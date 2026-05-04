use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

use anyhow::{Result, anyhow, bail};

use crate::{
    ast::{
        Ast, Command, CommandId, Def, Expr, ExprId, FuncSig, InfixOp, Program, Type, TypeContext,
        TypeId,
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
    Unbound,
    UnknownRecordField,
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
            TypecheckError::Unbound => write!(f, "Unbound variable"),
            TypecheckError::UnknownRecordField => write!(f, "Unknown record field"),
        }
    }
}

impl std::error::Error for TypecheckError {}

pub fn typecheck(program: &Program, ast: &mut Ast, tcx: &mut TypeContext) -> Result<()> {
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

fn check_def(def: &Def, env: &mut TypeEnv, ast: &Ast, tcx: &mut TypeContext) -> Result<()> {
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
            env.with_scope(|env| -> Result<()> {
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

fn check_pipeline(enabled: bool, body: CommandId, ast: &Ast) -> Result<()> {
    match &ast.commands[body] {
        Command::Seq(..) if enabled => Err(anyhow!(TypecheckError::PipelineError)),
        _ => Ok(()),
    }
}

fn check_command(
    cmd: CommandId,
    env: &mut TypeEnv,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<()> {
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
                bail!(TypecheckError::UnexpectedType);
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
                bail!(TypecheckError::UnexpectedType);
            }

            env.with_scope(|env| check_command(*body, env, ast, tcx))?;

            Ok(())
        }
        Command::Update { lhs, op: _op, rhs } => {
            let lhs_ty = check_expr(*lhs, env, ast, tcx)?;
            let rhs_ty = check_expr(*rhs, env, ast, tcx)?;

            if !is_subtype(rhs_ty, lhs_ty, tcx) {
                bail!(TypecheckError::UnexpectedType);
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
                            _ => bail!(TypecheckError::UnexpectedType),
                        };

                        if dims_len != 1 {
                            bail!(TypecheckError::Unsupported(
                                "Multidimensional array literals",
                            ));
                        }

                        if first_dim_len != vals.len(&ast.expr_lists) {
                            bail!(TypecheckError::LiteralLengthMismatch);
                        }

                        for val in vals.as_slice(&ast.expr_lists).iter() {
                            let val_ty = check_expr(*val, env, ast, tcx)?;
                            if !is_subtype(val_ty, element_type, tcx) {
                                bail!(TypecheckError::UnexpectedType);
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
                            _ => bail!(TypecheckError::UnexpectedType),
                        };

                        for (expected_id, expected_ty) in expected_fields {
                            let actual_ty = actual_types
                                .remove(expected_id)
                                .ok_or(TypecheckError::MissingField)?;

                            if !is_subtype(actual_ty, *expected_ty, tcx) {
                                bail!(TypecheckError::UnexpectedType);
                            }
                        }

                        if !actual_types.is_empty() {
                            bail!(TypecheckError::ExtraFields);
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
                                bail!(TypecheckError::UnexpectedType);
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

fn check_expr_(
    expr: ExprId,
    env: &mut TypeEnv,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<TypeId> {
    match &ast.exprs[expr] {
        Expr::RationalLiteral(v) => Ok(tcx.get_rational(v.clone())),
        Expr::IntLiteral { value, .. } => Ok(tcx.get_static_int(*value)),
        Expr::BoolLiteral(_) => Ok(tcx.get_bool()),
        Expr::RecordLiteral(..) | Expr::ArrayLiteral(..) => {
            Err(anyhow!(TypecheckError::NotInBinder))
        }
        Expr::Cast { expr, ty: cast_ty } => {
            check_expr(*expr, env, ast, tcx)?;
            // TODO: safe cast check
            Ok(*cast_ty)
        }
        Expr::Id(id) => {
            let ty = env.get(id).ok_or(TypecheckError::Unbound)?;
            tcx.id_type_map.insert(*id, ty);
            Ok(ty)
        }
        Expr::BinOp { left, op, right } => {
            let t1 = check_expr(*left, env, ast, tcx)?;
            let t2 = check_expr(*right, env, ast, tcx)?;
            Ok(check_binop(t1, t2, op.clone(), ast, tcx)?)
        }
        Expr::Application { func, args } => {
            let func_ty = env.get(func).ok_or(TypecheckError::Unbound)?;

            let (arg_types, ret) = match &tcx.types[func_ty] {
                Type::Func {
                    args: arg_types,
                    ret,
                } => (*arg_types, *ret),
                _ => bail!(TypecheckError::UnexpectedType),
            };

            if arg_types.len(&tcx.type_lists) != args.len(&ast.expr_lists) {
                bail!(TypecheckError::ArgLengthMismatch);
            }

            for i in 0..args.len(&ast.expr_lists) {
                let expected_ty = arg_types.as_slice(&tcx.type_lists)[i];
                let arg_ty = check_expr(args.as_slice(&ast.expr_lists)[i], env, ast, tcx)?;
                if !is_subtype(arg_ty, expected_ty, tcx) {
                    bail!(TypecheckError::UnexpectedType);
                }
            }

            Ok(ret)
        }
        Expr::RecordAccess { record, field } => {
            let record_ty = check_expr(*record, env, ast, tcx)?;

            let fields = match &tcx.types[record_ty] {
                Type::RecType { fields, .. } => fields,
                _ => bail!(TypecheckError::UnexpectedType),
            };

            fields
                .get(field)
                .copied()
                .ok_or_else(|| anyhow!(TypecheckError::UnknownRecordField))
        }
        Expr::ArrayAccess { array, indices } => {
            let (element_ty, dims_len) =
                match &tcx.types[env.get(array).ok_or(TypecheckError::Unbound)?] {
                    Type::Array {
                        element_type, dims, ..
                    } => (*element_type, dims.len()),
                    _ => bail!(TypecheckError::UnexpectedType),
                };

            if indices.len(&ast.expr_lists) != dims_len {
                bail!(TypecheckError::IncorrectAccessDims);
            }

            indices
                .as_slice(&ast.expr_lists)
                .iter()
                .try_for_each(|idx| -> Result<()> {
                    let idx_ty = check_expr(*idx, env, ast, tcx)?;
                    match &tcx.types[idx_ty] {
                        Type::StaticInt(..) | Type::Bit { .. } | Type::Index { .. } => Ok(()),
                        _ => Err(anyhow!(TypecheckError::UnexpectedType)),
                    }
                })?;

            tcx.id_type_map.insert(*array, element_ty);
            Ok(element_ty)
        }
    }
}

fn check_expr(expr: ExprId, env: &mut TypeEnv, ast: &Ast, tcx: &mut TypeContext) -> Result<TypeId> {
    let ty = check_expr_(expr, env, ast, tcx)?;
    if let Some(prev_ty) = tcx.expr_type_map.get(expr) {
        assert!(*prev_ty == ty, "Expression type changed during checking");
    }
    tcx.expr_type_map.insert(expr, ty);
    Ok(ty)
}

fn check_binop(
    tid1: TypeId,
    tid2: TypeId,
    op: InfixOp,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<TypeId> {
    todo!()
}
