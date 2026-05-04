use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

use crate::{
    ast::{
        Ast, Command, CommandId, Def, Expr, ExprId, FuncSig, Program, Type, TypeContext, TypeId,
    },
    subtyping::{bits_needed, is_subtype},
    type_env::TypeEnv,
};

#[derive(Debug)]
pub enum TypecheckError {
    UnexpectedType,
    NoJoin,
    BinopError,
    NotInBinder,
    ArgLengthMismatch,
    IncorrectAccessDims,
    InvalidShrinkWidth,
    InvalidAlignFactor,
    PipelineError,
    MissingField,
    ExtraFields,
    InvalidSplitFactor,
    AlreadyBound,
    ExplicitTypeMissing,
    Unsupported(&'static str),
    LiteralLengthMismatch,
    UnknownAlias,
    InvalidArrayDims,
}

impl Display for TypecheckError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TypecheckError::UnexpectedType => write!(f, "Unexpected type"),
            TypecheckError::NoJoin => write!(f, "No common supertype found"),
            TypecheckError::BinopError => write!(f, "Invalid binary operation"),
            TypecheckError::NotInBinder => write!(f, "Expression should be in let binder"),
            TypecheckError::ArgLengthMismatch => write!(f, "Argument length mismatch"),
            TypecheckError::IncorrectAccessDims => {
                write!(f, "Incorrect number of dimensions for array access")
            }
            TypecheckError::InvalidShrinkWidth => write!(f, "Invalid shrink width"),
            TypecheckError::InvalidAlignFactor => write!(f, "Invalid align factor"),
            TypecheckError::PipelineError => write!(f, "Pipeline error"),
            TypecheckError::MissingField => write!(f, "Missing field in struct literal"),
            TypecheckError::ExtraFields => write!(f, "Extra fields in struct literal"),
            TypecheckError::InvalidSplitFactor => write!(f, "Invalid split factor"),
            TypecheckError::AlreadyBound => write!(f, "Type is already bound"),
            TypecheckError::ExplicitTypeMissing => {
                write!(f, "Explicit type annotation is required")
            }
            TypecheckError::Unsupported(feature) => write!(f, "Unsupported feature: {}", feature),
            TypecheckError::LiteralLengthMismatch => write!(f, "Array literal length mismatch"),
            TypecheckError::UnknownAlias => write!(f, "Unknown type alias"),
            TypecheckError::InvalidArrayDims => write!(f, "Invalid array dimensions"),
        }
    }
}

impl std::error::Error for TypecheckError {}

pub fn typecheck(
    program: &Program,
    ast: &mut Ast,
    tcx: &mut TypeContext,
) -> Result<(), TypecheckError> {
    let mut env = TypeEnv::new();

    let all_defs: Vec<_> = program
        .includes
        .iter()
        .flat_map(|include| &include.defs)
        .chain(&program.defs)
        .collect();

    for def in all_defs {
        check_def(def, &mut env, ast, tcx)?;
    }

    // check the main command
    check_def(
        &Def::Func {
            sig: FuncSig {
                name: ast.ids.push("main".to_string()),
                args: vec![],
                ret_ty: tcx.get_void(),
            },
            body: program.cmd,
        },
        &mut env,
        ast,
        tcx,
    )?;

    Ok(())
}

fn check_def(
    def: &Def,
    env: &mut TypeEnv,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<(), TypecheckError> {
    match def {
        Def::Record { name, fields } => {
            let resolved_fields = fields
                .iter()
                .map(|(id, ty)| {
                    env.resolve_type(*ty, tcx)
                        .map(|resolved| (id.clone(), resolved))
                        .map_err(|_| TypecheckError::UnknownAlias)
                })
                .collect::<Result<_, _>>()?;
            env.add_type(
                name.clone(),
                tcx.get_rec_type(name.clone(), resolved_fields),
            )
            .map_err(|_| TypecheckError::AlreadyBound)?;
        }
        Def::Func { sig, body } => {
            env.with_scope(|env| {
                // add args to env
                for decl in &sig.args {
                    let resolved_ty = env
                        .resolve_type(decl.ty, tcx)
                        .map_err(|_| TypecheckError::UnknownAlias)?;
                    tcx.id_type_map.insert(decl.id, resolved_ty);
                    env.add(decl.id, resolved_ty, tcx)
                        .map_err(|_| TypecheckError::AlreadyBound)?;
                }

                // add return type to env
                let resolved_ret_ty = env
                    .resolve_type(sig.ret_ty, tcx)
                    .map_err(|_| TypecheckError::UnknownAlias)?;
                env.set_ret_type(resolved_ret_ty);

                check_command(*body, env, ast, tcx)?;
                Ok(())
            })?;

            // add function type to env
            // should add to id map as well?
            env.add(
                sig.name,
                tcx.get_func(sig.args.iter().map(|decl| decl.ty).collect(), sig.ret_ty),
                tcx,
            )
            .map_err(|_| TypecheckError::AlreadyBound)?;
        }
    }
    Ok(())
}

fn check_pipeline(enabled: bool, body: CommandId, ast: &Ast) -> Result<(), TypecheckError> {
    match &ast.commands[body] {
        Command::Seq(..) if enabled => Err(TypecheckError::PipelineError),
        _ => Ok(()),
    }
}

fn check_command(
    cmd: CommandId,
    env: &mut TypeEnv,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<(), TypecheckError> {
    match &ast.commands[cmd] {
        Command::Empty => Ok(()),
        Command::Par(cmds) | Command::Seq(cmds) => {
            for cmd in cmds {
                check_command(*cmd, env, ast, tcx)?;
            }
            Ok(())
        }
        Command::IfElse { cond, then, else_ } => {
            let cond_ty = check_expr(*cond, env, ast, tcx)?;
            if cond_ty != tcx.get_bool() {
                return Err(TypecheckError::UnexpectedType);
            }

            env.with_scope(|env| check_command(*then, env, ast, tcx))?;
            env.with_scope(|env| check_command(*else_, env, ast, tcx))?;

            Ok(())
        }
        Command::For {
            range,
            pipeline,
            body,
            combine,
        } => {
            check_pipeline(*pipeline, *body, ast)?;

            env.with_scope(|env| {
                env.add(
                    range.id,
                    tcx.get_index(
                        (0, range.unroll),
                        (range.start / range.unroll, range.end / range.unroll),
                    ),
                    tcx,
                )
                .map_err(|_| TypecheckError::AlreadyBound)?;

                check_command(*body, env, ast, tcx)?;
                check_command(*combine, env, ast, tcx)
            })?;

            Ok(())
        }
        Command::While {
            cond,
            pipeline,
            body,
        } => {
            check_pipeline(*pipeline, *body, ast)?;

            let cond_ty = check_expr(*cond, env, ast, tcx)?;
            if cond_ty != tcx.get_bool() {
                return Err(TypecheckError::UnexpectedType);
            }

            env.with_scope(|env| check_command(*body, env, ast, tcx))?;

            Ok(())
        }
        Command::Update { lhs, op: _op, rhs } => {
            let lhs_ty = check_expr(*lhs, env, ast, tcx)?;
            let rhs_ty = check_expr(*rhs, env, ast, tcx)?;

            if !is_subtype(rhs_ty, lhs_ty, tcx) {
                return Err(TypecheckError::UnexpectedType);
            }
            Ok(())
        }
        Command::Let { id, ty, value } => {
            if let Some(value) = value {
                match &ast.exprs[*value] {
                    Expr::ArrayLiteral(vals) => {
                        let ty = ty.ok_or(TypecheckError::ExplicitTypeMissing)?;
                        let resolved_ty = env
                            .resolve_type(ty, tcx)
                            .map_err(|_| TypecheckError::UnknownAlias)?;

                        let (element_type, dims_len, first_dim_len) = match &tcx.types[resolved_ty]
                        {
                            Type::Array {
                                element_type, dims, ..
                            } => (
                                *element_type,
                                dims.len(),
                                dims.first()
                                    .map(|d| d.length)
                                    .ok_or(TypecheckError::InvalidArrayDims)?,
                            ),
                            _ => return Err(TypecheckError::UnexpectedType),
                        };

                        if dims_len != 1 {
                            return Err(TypecheckError::Unsupported(
                                "Multidimensional array literals",
                            ));
                        }

                        if first_dim_len != vals.len(&ast.expr_lists) {
                            return Err(TypecheckError::LiteralLengthMismatch);
                        }

                        for val in vals.as_slice(&ast.expr_lists).iter() {
                            let val_ty = check_expr(*val, env, ast, tcx)?;
                            if !is_subtype(val_ty, element_type, tcx) {
                                return Err(TypecheckError::UnexpectedType);
                            }
                        }

                        tcx.id_type_map.insert(*id, ty);
                        env.add(*id, resolved_ty, tcx)
                            .map_err(|_| TypecheckError::AlreadyBound)?;
                    }

                    Expr::RecordLiteral(fields) => {
                        let ty = ty.ok_or(TypecheckError::ExplicitTypeMissing)?;
                        let resolved_ty = env
                            .resolve_type(ty, tcx)
                            .map_err(|_| TypecheckError::UnknownAlias)?;

                        let mut actual_types = HashMap::new();
                        for (id, expr) in fields {
                            let field_ty = check_expr(*expr, env, ast, tcx)?;
                            actual_types.insert(*id, field_ty);
                        }

                        let expected_fields = match &tcx.types[resolved_ty] {
                            Type::RecType { fields, .. } => fields,
                            _ => return Err(TypecheckError::UnexpectedType),
                        };

                        for (expected_id, expected_ty) in expected_fields {
                            let actual_ty = actual_types
                                .remove(expected_id)
                                .ok_or(TypecheckError::MissingField)?;

                            if !is_subtype(actual_ty, *expected_ty, tcx) {
                                return Err(TypecheckError::UnexpectedType);
                            }
                        }

                        if !actual_types.is_empty() {
                            return Err(TypecheckError::ExtraFields);
                        }

                        env.add(*id, resolved_ty, tcx)
                            .map_err(|_| TypecheckError::AlreadyBound)?;
                    }
                    _ => {
                        let resolved_ty = ty
                            .map(|ty| {
                                env.resolve_type(ty, tcx)
                                    .map_err(|_| TypecheckError::UnknownAlias)
                            })
                            .transpose()?;

                        let val_ty = check_expr(*value, env, ast, tcx)?;

                        if let Some(resolved_ty) = resolved_ty {
                            let resolved_ty = match &tcx.types[resolved_ty] {
                                Type::StaticInt(v) => tcx.get_bit(bits_needed(*v), false),
                                Type::Rational(_) => tcx.get_double(),
                                _ => resolved_ty,
                            };

                            if !is_subtype(val_ty, resolved_ty, tcx) {
                                return Err(TypecheckError::UnexpectedType);
                            }
                            env.add(*id, resolved_ty, tcx)
                                .map_err(|_| TypecheckError::AlreadyBound)?;
                        } else {
                            let typ = match &tcx.types[val_ty] {
                                Type::StaticInt(v) => tcx.get_bit(bits_needed(*v), false),
                                Type::Rational(_) => tcx.get_double(),
                                _ => val_ty,
                            };

                            env.add(*id, typ, tcx)
                                .map_err(|_| TypecheckError::AlreadyBound)?;
                        }
                    }
                }
            } else {
                let ty = ty.ok_or(TypecheckError::ExplicitTypeMissing)?;
                let resolved_ty = env
                    .resolve_type(ty, tcx)
                    .map_err(|_| TypecheckError::UnknownAlias)?;
                env.add(*id, resolved_ty, tcx)
                    .map_err(|_| TypecheckError::AlreadyBound)?;
            }

            Ok(())
        }
        _ => todo!(),
    }
}

fn check_expr(
    expr: ExprId,
    env: &mut TypeEnv,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<TypeId, TypecheckError> {
    match &ast.exprs[expr] {
        _ => todo!(),
    }
}
