use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use thiserror::Error;

use crate::{
    ast::{
        Ast, Command, CommandId, Def, Expr, ExprId, FuncSig, InfixOp, Program, Type, TypeContext,
        TypeId,
    },
    subtyping::is_subtype,
    type_env::TypeEnv,
    utils::{bits_needed, resolve_id},
};

#[derive(Debug, Error)]
pub enum TypecheckError {
    #[error("Unexpected type")]
    UnexpectedType,
    #[error("No common supertype found")]
    NoJoin,
    #[error("Invalid binary operation")]
    BinopError,
    #[error("Expression should be in let binder")]
    NotInBinder,
    #[error("Argument length mismatch")]
    ArgLengthMismatch,
    #[error("Incorrect number of dimensions for array access")]
    IncorrectAccessDims,
    #[error("Invalid shrink width")]
    InvalidShrinkWidth,
    #[error("Invalid align factor")]
    InvalidAlignFactor,
    #[error("Pipeline error")]
    PipelineError,
    #[error("Missing field in struct literal")]
    MissingField,
    #[error("Extra fields in struct literal")]
    ExtraFields,
    #[error("Invalid split factor")]
    InvalidSplitFactor,
    #[error("Type is already bound")]
    AlreadyBound,
    #[error("Explicit type annotation is required")]
    ExplicitTypeMissing,
    #[error("Unsupported feature: {0}")]
    Unsupported(&'static str),
    #[error("Array literal length mismatch")]
    LiteralLengthMismatch,
    #[error("Unknown type alias")]
    UnknownAlias,
    #[error("Invalid array dimensions")]
    InvalidArrayDims,
    #[error("Unbound variable")]
    Unbound,
    #[error("Unknown record field")]
    UnknownRecordField,
}

pub fn typecheck(program: &Program, ast: &mut Ast, tcx: &mut TypeContext) -> Result<()> {
    let mut env = TypeEnv::new();

    let all_defs: Vec<_> = program
        .includes
        .iter()
        .flat_map(|include| &include.defs)
        .chain(&program.defs)
        .collect();

    for def in all_defs {
        check_def(def, &mut env, ast, tcx).with_context(|| {
            format!(
                "failed to type check definition `{}`",
                match def {
                    Def::Record { name, .. } => resolve_id(*name, ast),
                    Def::Func { sig, .. } => resolve_id(sig.name, ast),
                }
            )
        })?;
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
    )
    .context("failed to type check the main command")?;

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
                .collect::<Result<_, _>>()
                .with_context(|| {
                    format!(
                        "failed to type check record definition `{}` field types",
                        resolve_id(*name, ast)
                    )
                })?;
            env.add_type(
                name.clone(),
                tcx.get_rec_type(name.clone(), resolved_fields),
            )
            .map_err(|_| TypecheckError::AlreadyBound)
            .with_context(|| format!("record type `{}` already bound", resolve_id(*name, ast)))?;
        }
        Def::Func { sig, body } => {
            env.with_scope(|env| -> Result<()> {
                // add args to env
                for decl in &sig.args {
                    let resolved_ty = env
                        .resolve_type(decl.ty, tcx)
                        .map_err(|_| TypecheckError::UnknownAlias)
                        .with_context(|| {
                            format!(
                                "failed to type check function `{}` argument `{}` type",
                                resolve_id(sig.name, ast),
                                resolve_id(decl.id, ast),
                            )
                        })?;
                    tcx.id_type_map.insert(decl.id, resolved_ty);
                    env.add(decl.id, resolved_ty, tcx)
                        .map_err(|_| TypecheckError::AlreadyBound)
                        .with_context(|| {
                            format!(
                                "argument `{}` for function `{}` already bound",
                                resolve_id(decl.id, ast),
                                resolve_id(sig.name, ast)
                            )
                        })?;
                }

                // add return type to env
                let resolved_ret_ty = env
                    .resolve_type(sig.ret_ty, tcx)
                    .map_err(|_| TypecheckError::UnknownAlias)
                    .with_context(|| {
                        format!(
                            "failed to type check function `{}` return type",
                            resolve_id(sig.name, ast)
                        )
                    })?;
                env.set_ret_type(resolved_ret_ty);

                check_command(*body, env, ast, tcx).with_context(|| {
                    format!(
                        "failed to type check command: function `{}` body",
                        resolve_id(sig.name, ast)
                    )
                })?;
                Ok(())
            })?;

            // add function type to env
            // should add to id map as well?
            env.add(
                sig.name,
                tcx.get_func(sig.args.iter().map(|decl| decl.ty).collect(), sig.ret_ty),
                tcx,
            )
            .map_err(|_| TypecheckError::AlreadyBound)
            .with_context(|| format!("function `{}` already bound", resolve_id(sig.name, ast)))?;
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
        Command::Par(cmds) => {
            for cmd in cmds {
                check_command(*cmd, env, ast, tcx)
                    .context("failed to type check command: par block")?;
            }
            Ok(())
        }
        Command::Seq(cmds) => {
            for cmd in cmds {
                check_command(*cmd, env, ast, tcx)
                    .context("failed to type check command: seq block")?;
            }
            Ok(())
        }
        Command::IfElse { cond, then, else_ } => {
            let cond_ty = check_expr(*cond, env, ast, tcx)
                .context("failed to type check expression: if condition")?;
            if cond_ty != tcx.get_bool() {
                Err(anyhow!(TypecheckError::UnexpectedType))
                    .context("failed to type check expression: if condition must be bool")?;
            }

            env.with_scope(|env| check_command(*then, env, ast, tcx))
                .context("failed to type check command: if then branch")?;
            env.with_scope(|env| check_command(*else_, env, ast, tcx))
                .context("failed to type check command: if else branch")?;

            Ok(())
        }
        Command::For {
            range,
            pipeline,
            body,
            combine,
        } => {
            check_pipeline(*pipeline, *body, ast)
                .context("failed to type check command: for pipeline")?;

            env.with_scope(|env| {
                env.add(
                    range.id,
                    tcx.get_index(
                        (0, range.unroll),
                        (range.start / range.unroll, range.end / range.unroll),
                    ),
                    tcx,
                )
                .map_err(|_| TypecheckError::AlreadyBound)
                .with_context(|| {
                    format!(
                        "for range iterator `{}` already bound",
                        resolve_id(range.id, ast)
                    )
                })?;

                check_command(*body, env, ast, tcx)
                    .context("failed to type check command: for body")?;
                check_command(*combine, env, ast, tcx)
                    .context("failed to type check command: for combine")
            })?;

            Ok(())
        }
        Command::While {
            cond,
            pipeline,
            body,
        } => {
            check_pipeline(*pipeline, *body, ast)
                .context("failed to type check command: while pipeline")?;

            let cond_ty = check_expr(*cond, env, ast, tcx)
                .context("failed to type check expression: while condition")?;
            if cond_ty != tcx.get_bool() {
                Err(anyhow!(TypecheckError::UnexpectedType))
                    .context("failed to type check expression: while condition must be bool")?;
            }

            env.with_scope(|env| check_command(*body, env, ast, tcx))
                .context("failed to type check command: while body")?;

            Ok(())
        }
        Command::Update { lhs, op: _op, rhs } => {
            let lhs_ty = check_expr(*lhs, env, ast, tcx)
                .context("failed to type check expression: update LHS")?;
            let rhs_ty = check_expr(*rhs, env, ast, tcx)
                .context("failed to type check expression: update RHS")?;

            if !is_subtype(rhs_ty, lhs_ty, tcx) {
                Err(anyhow!(TypecheckError::UnexpectedType))
                    .context("failed to type check command: update")?;
            }
            Ok(())
        }
        Command::Let { id, ty, value } => {
            if let Some(value) = value {
                match &ast.exprs[*value] {
                    Expr::ArrayLiteral(vals) => {
                        let ty =
                            ty.ok_or(TypecheckError::ExplicitTypeMissing)
                                .with_context(|| {
                                    format!(
                                        "failed to type check command: let binding `{}` missing explicit type",
                                        resolve_id(*id, ast)
                                    )
                                })?;
                        let resolved_ty = env
                            .resolve_type(ty, tcx)
                            .map_err(|_| TypecheckError::UnknownAlias)
                            .with_context(|| {
                                format!(
                                    "failed to type check command: let binding `{}` type annotation",
                                    resolve_id(*id, ast)
                                )
                            })?;

                        let (element_type, dims_len, first_dim_len) = match &tcx.types[resolved_ty]
                        {
                            Type::Array {
                                element_type, dims, ..
                            } => (
                                *element_type,
                                dims.len(),
                                dims.first()
                                    .map(|d| d.length)
                                    .ok_or(TypecheckError::InvalidArrayDims)
                                    .with_context(|| {
                                        format!(
                                            "failed to type check expression: array literal for let binding `{}`",
                                            resolve_id(*id, ast)
                                        )
                                    })?,
                            ),
                            _ => Err(anyhow!(TypecheckError::UnexpectedType)).with_context(
                                || {
                                    format!(
                                        "failed to type check expression: array literal for let binding `{}`",
                                        resolve_id(*id, ast)
                                    )
                                },
                            )?,
                        };

                        if dims_len != 1 {
                            Err(anyhow!(TypecheckError::Unsupported(
                                "Multidimensional array literals",
                            )))
                            .with_context(|| {
                                format!(
                                    "failed to type check expression: array literal for let binding `{}`",
                                    resolve_id(*id, ast)
                                )
                            })?;
                        }

                        if first_dim_len != vals.len(&ast.expr_lists) {
                            Err(anyhow!(TypecheckError::LiteralLengthMismatch)).with_context(
                                || {
                                    format!(
                                        "failed to type check expression: array literal for let binding `{}`",
                                        resolve_id(*id, ast)
                                    )
                                },
                            )?;
                        }

                        for val in vals.as_slice(&ast.expr_lists).iter() {
                            let val_ty = check_expr(*val, env, ast, tcx).with_context(|| {
                                format!(
                                    "failed to type check expression: array element for let binding `{}`",
                                    resolve_id(*id, ast)
                                )
                            })?;
                            if !is_subtype(val_ty, element_type, tcx) {
                                Err(anyhow!(TypecheckError::UnexpectedType)).with_context(
                                    || {
                                        format!(
                                            "failed to type check expression: array element for let binding `{}`",
                                            resolve_id(*id, ast)
                                        )
                                    },
                                )?;
                            }
                        }

                        tcx.id_type_map.insert(*id, ty);
                        env.add(*id, resolved_ty, tcx)
                            .map_err(|_| TypecheckError::AlreadyBound)
                            .with_context(|| format!("`{}` already bound", resolve_id(*id, ast)))?;
                    }

                    Expr::RecordLiteral(fields) => {
                        let ty =
                            ty.ok_or(TypecheckError::ExplicitTypeMissing)
                                .with_context(|| {
                                    format!(
                                        "failed to type check command: let binding `{}` missing explicit type",
                                        resolve_id(*id, ast)
                                    )
                                })?;
                        let resolved_ty = env
                            .resolve_type(ty, tcx)
                            .map_err(|_| TypecheckError::UnknownAlias)
                            .with_context(|| {
                                format!(
                                    "failed to type check command: let binding `{}` type annotation",
                                    resolve_id(*id, ast)
                                )
                            })?;

                        let mut actual_types = HashMap::new();
                        for (field_id, expr) in fields {
                            let field_ty = check_expr(*expr, env, ast, tcx).with_context(|| {
                                format!(
                                    "failed to type check expression: record field `{}` for let binding `{}`",
                                    resolve_id(*field_id, ast),
                                    resolve_id(*id, ast)
                                )
                            })?;
                            actual_types.insert(*field_id, field_ty);
                        }

                        let expected_fields = match &tcx.types[resolved_ty] {
                            Type::RecType { fields, .. } => fields,
                            _ => Err(anyhow!(TypecheckError::UnexpectedType)).with_context(
                                || {
                                    format!(
                                        "failed to type check expression: record literal for let binding `{}`",
                                        resolve_id(*id, ast)
                                    )
                                },
                            )?,
                        };

                        for (expected_id, expected_ty) in expected_fields {
                            let actual_ty = actual_types
                                .remove(expected_id)
                                .ok_or(TypecheckError::MissingField)
                                .with_context(|| {
                                    format!(
                                        "failed to type check expression: record literal missing field `{}` for let binding `{}`",
                                        resolve_id(*expected_id, ast),
                                        resolve_id(*id, ast),
                                    )
                                })?;

                            if !is_subtype(actual_ty, *expected_ty, tcx) {
                                Err(anyhow!(TypecheckError::UnexpectedType)).with_context(
                                    || {
                                        format!(
                                            "failed to type check expression: record field `{}` for let binding `{}`",
                                            resolve_id(*expected_id, ast),
                                            resolve_id(*id, ast)
                                        )
                                    },
                                )?;
                            }
                        }

                        if !actual_types.is_empty() {
                            Err(anyhow!(TypecheckError::ExtraFields)).with_context(|| {
                                format!(
                                    "failed to type check expression: record literal for let binding `{}`",
                                    resolve_id(*id, ast)
                                )
                            })?;
                        }

                        env.add(*id, resolved_ty, tcx)
                            .map_err(|_| TypecheckError::AlreadyBound)
                            .with_context(|| format!("`{}` already bound", resolve_id(*id, ast)))?;
                    }
                    _ => {
                        let resolved_ty = ty
                            .map(|ty| {
                                env.resolve_type(ty, tcx)
                                    .map_err(|_| TypecheckError::UnknownAlias)
                            })
                            .transpose()
                            .with_context(|| {
                                format!(
                                    "failed to type check command: let binding `{}` type annotation",
                                    resolve_id(*id, ast)
                                )
                            })?;

                        let val_ty = check_expr(*value, env, ast, tcx).with_context(|| {
                            format!(
                                "failed to type check expression: value for let binding `{}`",
                                resolve_id(*id, ast)
                            )
                        })?;

                        if let Some(resolved_ty) = resolved_ty {
                            let resolved_ty = match &tcx.types[resolved_ty] {
                                Type::StaticInt(v) => tcx.get_bit(bits_needed(*v), false),
                                Type::Rational(_) => tcx.get_double(),
                                _ => resolved_ty,
                            };

                            if !is_subtype(val_ty, resolved_ty, tcx) {
                                Err(anyhow!(TypecheckError::UnexpectedType)).with_context(
                                    || {
                                        format!(
                                            "failed to type check expression: value for let binding `{}`",
                                            resolve_id(*id, ast)
                                        )
                                    },
                                )?;
                            }
                            env.add(*id, resolved_ty, tcx)
                                .map_err(|_| TypecheckError::AlreadyBound)
                                .with_context(|| {
                                    format!("`{}` already bound", resolve_id(*id, ast))
                                })?;
                        } else {
                            let typ = match &tcx.types[val_ty] {
                                Type::StaticInt(v) => tcx.get_bit(bits_needed(*v), false),
                                Type::Rational(_) => tcx.get_double(),
                                _ => val_ty,
                            };

                            env.add(*id, typ, tcx)
                                .map_err(|_| TypecheckError::AlreadyBound)
                                .with_context(|| {
                                    format!("`{}` already bound", resolve_id(*id, ast))
                                })?;
                        }
                    }
                }
            } else {
                let ty = ty
                    .ok_or(TypecheckError::ExplicitTypeMissing)
                    .with_context(|| {
                        format!(
                            "failed to type check command: let binding `{}` missing explicit type",
                            resolve_id(*id, ast)
                        )
                    })?;
                let resolved_ty = env
                    .resolve_type(ty, tcx)
                    .map_err(|_| TypecheckError::UnknownAlias)
                    .with_context(|| {
                        format!(
                            "failed to type check command: let binding `{}` type annotation",
                            resolve_id(*id, ast)
                        )
                    })?;
                env.add(*id, resolved_ty, tcx)
                    .map_err(|_| TypecheckError::AlreadyBound)
                    .with_context(|| format!("`{}` already bound", resolve_id(*id, ast)))?;
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
            check_expr(*expr, env, ast, tcx)
                .context("failed to type check expression: cast operand")?;
            // TODO: safe cast check
            Ok(*cast_ty)
        }
        Expr::Id(id) => {
            let ty = env
                .get(id)
                .ok_or(TypecheckError::Unbound)
                .with_context(|| format!("`{}` unbound", resolve_id(*id, ast)))?;
            tcx.id_type_map.insert(*id, ty);
            Ok(ty)
        }
        Expr::BinOp { left, op, right } => {
            let t1 = check_expr(*left, env, ast, tcx)
                .context("failed to type check expression: binary operator LHS")?;
            let t2 = check_expr(*right, env, ast, tcx)
                .context("failed to type check expression: binary operator RHS")?;
            Ok(check_binop(t1, t2, op.clone(), ast, tcx)
                .context("failed to type check binary operation")?)
        }
        Expr::Application { func, args } => {
            let func_ty = env
                .get(func)
                .ok_or(TypecheckError::Unbound)
                .with_context(|| {
                    format!(
                        "failed to type check expression: function application `{}`",
                        resolve_id(*func, ast)
                    )
                })?;

            let (arg_types, ret) = match &tcx.types[func_ty] {
                Type::Func {
                    args: arg_types,
                    ret,
                } => (*arg_types, *ret),
                _ => Err(anyhow!(TypecheckError::UnexpectedType))
                    .context("failed to type check expression: function application callee")?,
            };

            if arg_types.len(&tcx.type_lists) != args.len(&ast.expr_lists) {
                Err(anyhow!(TypecheckError::ArgLengthMismatch))
                    .context("failed to type check expression: function application arguments")?;
            }

            for i in 0..args.len(&ast.expr_lists) {
                let expected_ty = arg_types.as_slice(&tcx.type_lists)[i];
                let arg_ty = check_expr(args.as_slice(&ast.expr_lists)[i], env, ast, tcx)
                    .with_context(|| {
                        format!(
                            "failed to type check expression: function application argument {}",
                            i
                        )
                    })?;
                if !is_subtype(arg_ty, expected_ty, tcx) {
                    Err(anyhow!(TypecheckError::UnexpectedType)).with_context(|| {
                        format!(
                            "failed to type check expression: function application argument {}",
                            i
                        )
                    })?;
                }
            }

            Ok(ret)
        }
        Expr::RecordAccess { record, field } => {
            let record_ty = check_expr(*record, env, ast, tcx)
                .context("failed to type check expression: record access")?;

            let fields = match &tcx.types[record_ty] {
                Type::RecType { fields, .. } => fields,
                _ => Err(anyhow!(TypecheckError::UnexpectedType))
                    .context("failed to type check expression: record access")?,
            };

            fields
                .get(field)
                .copied()
                .ok_or_else(|| anyhow!(TypecheckError::UnknownRecordField))
                .with_context(|| {
                    format!(
                        "failed to type check expression: record field `{}`",
                        resolve_id(*field, ast)
                    )
                })
        }
        Expr::ArrayAccess { array, indices } => {
            let (element_ty, dims_len) = match &tcx.types[env
                .get(array)
                .ok_or(TypecheckError::Unbound)
                .with_context(|| {
                    format!(
                        "failed to type check expression: array access `{}`",
                        resolve_id(*array, ast)
                    )
                })?] {
                Type::Array {
                    element_type, dims, ..
                } => (*element_type, dims.len()),
                _ => Err(anyhow!(TypecheckError::UnexpectedType)).with_context(|| {
                    format!(
                        "failed to type check expression: array access `{}`",
                        resolve_id(*array, ast)
                    )
                })?,
            };

            if indices.len(&ast.expr_lists) != dims_len {
                Err(anyhow!(TypecheckError::IncorrectAccessDims)).with_context(|| {
                    format!(
                        "failed to type check expression: array access `{}` indices",
                        resolve_id(*array, ast)
                    )
                })?;
            }

            indices
                .as_slice(&ast.expr_lists)
                .iter()
                .try_for_each(|idx| -> Result<()> {
                    let idx_ty = check_expr(*idx, env, ast, tcx)
                        .context("failed to type check expression: array index")?;
                    match &tcx.types[idx_ty] {
                        Type::StaticInt(..) | Type::Bit { .. } | Type::Index { .. } => Ok(()),
                        _ => Err(anyhow!(TypecheckError::UnexpectedType))
                            .context("failed to type check expression: array index"),
                    }
                })
                .with_context(|| {
                    format!(
                        "failed to type check expression: array access `{}` indices",
                        resolve_id(*array, ast)
                    )
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
