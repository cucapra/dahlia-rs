use std::collections::HashMap;

use crate::errors::TypecheckError;
use anyhow::{Context, Result, anyhow};

use crate::{
    ast::{
        Ast, Command, CommandId, Decl, Def, DimSpec, Expr, ExprId, IdResolve, InfixOp, Program,
        Suffix, Type, TypeContext, TypeId, View,
    },
    subtyping::{is_subtype, join_of},
    type_env::TypeEnv,
    utils::bits_needed,
};

pub fn typecheck(program: &Program, context: &mut crate::ast::Context) -> Result<()> {
    let ast = &mut context.ast;
    let tcx = &mut context.tcx;
    let mut env = TypeEnv::new();

    program
        .includes
        .iter()
        .flat_map(|include| &include.defs)
        .chain(&program.defs)
        .try_for_each(|def| {
            check_def(def, &mut env, ast, tcx).with_context(|| {
                format!(
                    "failed to type check definition `{}`",
                    match def {
                        Def::Record { name, .. } => name.resolve_id(ast),
                        Def::Func { sig, .. } => sig.name.resolve_id(ast),
                    }
                )
            })
        })?;

    for decl in &program.decls {
        check_decl(decl, &mut env, ast, tcx).with_context(|| {
            format!(
                "failed to type check program declaration `{}`",
                decl.id.resolve_id(ast)
            )
        })?;
    }

    // check the main command
    env.set_ret_type(tcx.get_void());
    check_command(program.cmd, &mut env, ast, tcx)
        .context("failed to type check the main command")?;

    Ok(())
}

fn check_decl(decl: &Decl, env: &mut TypeEnv, ast: &Ast, tcx: &mut TypeContext) -> Result<()> {
    let resolved_ty = env
        .resolve_type(decl.ty, tcx)
        .map_err(|_| TypecheckError::UnknownAlias)
        .with_context(|| {
            format!(
                "failed to type check declaration `{}` type",
                decl.id.resolve_id(ast),
            )
        })?;
    tcx.value_type_map.insert(decl.id, resolved_ty);
    Ok(())
}

fn check_def(def: &Def, env: &mut TypeEnv, ast: &Ast, tcx: &mut TypeContext) -> Result<()> {
    match def {
        Def::Record { name, fields } => {
            let resolved_fields = fields
                .iter()
                .map(|(id, ty)| {
                    env.resolve_type(*ty, tcx)
                        .map(|resolved| (*id, resolved))
                        .map_err(|_| TypecheckError::UnknownAlias)
                        .with_context(|| {
                            format!(
                                "failed to type check record definition `{}` field `{}` type",
                                name.resolve_id(ast),
                                id.resolve_id(ast)
                            )
                        })
                })
                .collect::<Result<_>>()?;
            env.add_record(*name, tcx.get_rec_type(*name, resolved_fields))
                .map_err(|_| TypecheckError::AlreadyBound)
                .with_context(|| format!("record type `{}` already bound", name.resolve_id(ast)))?;
        }
        Def::Func { sig, body } => {
            // add args to env
            for decl in &sig.args {
                check_decl(decl, env, ast, tcx).with_context(|| {
                    format!(
                        "failed to type check function `{}` argument `{}`",
                        sig.name.resolve_id(ast),
                        decl.id.resolve_id(ast),
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
                        sig.name.resolve_id(ast)
                    )
                })?;
            env.set_ret_type(resolved_ret_ty);

            check_command(*body, env, ast, tcx).with_context(|| {
                format!(
                    "failed to type check command: function `{}` body",
                    sig.name.resolve_id(ast)
                )
            })?;

            let resolved_arg_types = sig
                .args
                .iter()
                .map(|decl| {
                    env.resolve_type(decl.ty, tcx)
                        .map_err(|_| anyhow!(TypecheckError::UnknownAlias))
                        .with_context(|| {
                            format!(
                                "failed to type check function `{}` argument `{}` type",
                                sig.name.resolve_id(ast),
                                decl.id.resolve_id(ast),
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;

            let fn_type = tcx.get_func(resolved_arg_types, resolved_ret_ty);
            env.add_func(sig.name, fn_type)
                .map_err(|_| TypecheckError::AlreadyBound)
                .with_context(|| {
                    format!("function `{}` already bound", sig.name.resolve_id(ast))
                })?;
            tcx.func_type_map.insert(sig.name, fn_type);
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
        Command::Block(cmd) => {
            check_command(*cmd, env, ast, tcx).context("failed to type check command: block")?;

            Ok(())
        }
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

            check_command(*then, env, ast, tcx)
                .context("failed to type check command: if then branch")?;
            check_command(*else_, env, ast, tcx)
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

            let iter_ty = tcx.get_index(
                (0, range.unroll),
                (range.start / range.unroll, range.end / range.unroll),
            );
            tcx.value_type_map.insert(range.iter, iter_ty);

            check_command(*body, env, ast, tcx)
                .context("failed to type check command: for body")?;
            check_command(*combine, env, ast, tcx)
                .context("failed to type check command: for combine")?;

            Ok(())
        }
        Command::While {
            cond,
            pipeline,
            body,
        } => {
            check_pipeline(*pipeline, *body, ast).context(format!(
                "failed to type check command: while pipeline {}",
                *pipeline
            ))?;

            let cond_ty = check_expr(*cond, env, ast, tcx)
                .context("failed to type check expression: while condition")?;
            if cond_ty != tcx.get_bool() {
                Err(anyhow!(TypecheckError::UnexpectedType))
                    .context("failed to type check expression: while condition must be bool")?;
            }

            check_command(*body, env, ast, tcx)
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
                                        id.resolve_id(ast)
                                    )
                                })?;
                        let resolved_ty = env
                            .resolve_type(ty, tcx)
                            .map_err(|_| TypecheckError::UnknownAlias)
                            .with_context(|| {
                                format!(
                                    "failed to type check command: let binding `{}` type annotation",
                                    id.resolve_id(ast)
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
                                            id.resolve_id(ast)
                                        )
                                    })?,
                            ),
                            _ => Err(anyhow!(TypecheckError::UnexpectedType)).with_context(
                                || {
                                    format!(
                                        "failed to type check expression: array literal for let binding `{}`",
                                        id.resolve_id(ast)
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
                                    id.resolve_id(ast)
                                )
                            })?;
                        }

                        if first_dim_len != vals.len(&ast.expr_lists) {
                            Err(anyhow!(TypecheckError::LiteralLengthMismatch)).with_context(
                                || {
                                    format!(
                                        "failed to type check expression: array literal for let binding `{}`",
                                        id.resolve_id(ast)
                                    )
                                },
                            )?;
                        }

                        for val in vals.as_slice(&ast.expr_lists) {
                            let val_ty = check_expr(*val, env, ast, tcx).with_context(|| {
                                format!(
                                    "failed to type check expression: array element for let binding `{}`",
                                    id.resolve_id(ast)
                                )
                            })?;
                            if !is_subtype(val_ty, element_type, tcx) {
                                Err(anyhow!(TypecheckError::UnexpectedType)).with_context(
                                    || {
                                        format!(
                                            "failed to type check expression: array element for let binding `{}`",
                                            id.resolve_id(ast)
                                        )
                                    },
                                )?;
                            }
                        }

                        tcx.value_type_map.insert(*id, resolved_ty);
                    }

                    Expr::RecordLiteral(fields) => {
                        let ty =
                            ty.ok_or(TypecheckError::ExplicitTypeMissing)
                                .with_context(|| {
                                    format!(
                                        "failed to type check command: let binding `{}` missing explicit type",
                                        id.resolve_id(ast)
                                    )
                                })?;
                        let resolved_ty = env
                            .resolve_type(ty, tcx)
                            .map_err(|_| TypecheckError::UnknownAlias)
                            .with_context(|| {
                                format!(
                                    "failed to type check command: let binding `{}` type annotation",
                                    id.resolve_id(ast)
                                )
                            })?;

                        let mut actual_types = HashMap::new();
                        for (field_id, expr) in fields {
                            let field_ty = check_expr(*expr, env, ast, tcx).with_context(|| {
                                format!(
                                    "failed to type check expression: record field `{}` for let binding `{}`",
                                    field_id.resolve_id(ast),
                                    id.resolve_id(ast)
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
                                        id.resolve_id(ast)
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
                                        expected_id.resolve_id(ast),
                                        id.resolve_id(ast),
                                    )
                                })?;

                            if !is_subtype(actual_ty, *expected_ty, tcx) {
                                Err(anyhow!(TypecheckError::UnexpectedType)).with_context(
                                    || {
                                        format!(
                                            "failed to type check expression: record field `{}` for let binding `{}`",
                                            expected_id.resolve_id(ast),
                                            id.resolve_id(ast)
                                        )
                                    },
                                )?;
                            }
                        }

                        if !actual_types.is_empty() {
                            Err(anyhow!(TypecheckError::ExtraFields)).with_context(|| {
                                format!(
                                    "failed to type check expression: record literal for let binding `{}`",
                                    id.resolve_id(ast)
                                )
                            })?;
                        }

                        tcx.value_type_map.insert(*id, resolved_ty);
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
                                    id.resolve_id(ast)
                                )
                            })?;

                        let val_ty = check_expr(*value, env, ast, tcx).with_context(|| {
                            format!(
                                "failed to type check expression: value for let binding `{}`",
                                id.resolve_id(ast)
                            )
                        })?;

                        let val_ty = env
                            .resolve_type(val_ty, tcx)
                            .map_err(|_| TypecheckError::UnknownAlias)
                            .with_context(|| {
                                format!(
                                    "failed to type check expression: value for let binding `{}`",
                                    id.resolve_id(ast)
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
                                            id.resolve_id(ast)
                                        )
                                    },
                                )?;
                            }
                            tcx.value_type_map.insert(*id, resolved_ty);
                        } else {
                            let typ = match &tcx.types[val_ty] {
                                Type::StaticInt(v) => tcx.get_bit(bits_needed(*v), false),
                                Type::Rational(_) => tcx.get_double(),
                                _ => val_ty,
                            };

                            tcx.value_type_map.insert(*id, typ);
                        }
                    }
                }
            } else {
                let ty = ty
                    .ok_or(TypecheckError::ExplicitTypeMissing)
                    .with_context(|| {
                        format!(
                            "failed to type check command: let binding `{}` missing explicit type",
                            id.resolve_id(ast)
                        )
                    })?;
                let resolved_ty = env
                    .resolve_type(ty, tcx)
                    .map_err(|_| TypecheckError::UnknownAlias)
                    .with_context(|| {
                        format!(
                            "failed to type check command: let binding `{}` type annotation",
                            id.resolve_id(ast)
                        )
                    })?;
                tcx.value_type_map.insert(*id, resolved_ty);
            }

            Ok(())
        }
        Command::Decorate(..) => Ok(()),
        Command::Expr(expr) => {
            check_expr(*expr, env, ast, tcx)
                .context("failed to type check command: expression statement")?;
            Ok(())
        }
        Command::Return(expr) => {
            let ret_ty = env
                .get_ret_type()
                .expect("Return type not set in type environment");
            let expr_ty = check_expr(*expr, env, ast, tcx)
                .context("failed to type check command: return expression")?;
            if !is_subtype(expr_ty, ret_ty, tcx) {
                Err(anyhow!(TypecheckError::UnexpectedType))
                    .context("failed to type check command: return expression type mismatch")?;
            }
            Ok(())
        }
        Command::View {
            id,
            arr_id,
            dims: view_dims,
        } => {
            let arr_ty = tcx
                .value_type_map
                .get(arr_id)
                .copied()
                .expect("array should be bound after name resolution");

            let (element_ty, arr_dims_len, ports) = match &tcx.types[arr_ty] {
                Type::Array {
                    element_type,
                    dims,
                    ports,
                    ..
                } => (*element_type, dims.len(), *ports),
                _ => Err(anyhow!(TypecheckError::UnexpectedType)).with_context(|| {
                    format!(
                        "failed to type check command: view `{}`",
                        id.resolve_id(ast)
                    )
                })?,
            };

            if arr_dims_len != view_dims.len() {
                Err(anyhow!(TypecheckError::IncorrectAccessDims)).with_context(|| {
                    format!(
                        "failed to type check command: view `{}`",
                        id.resolve_id(ast)
                    )
                })?;
            }

            let mut new_dims = Vec::new();
            for i in 0..view_dims.len() {
                let arr_dim = match &tcx.types[arr_ty] {
                    Type::Array { dims, .. } => dims[i],
                    _ => unreachable!(),
                };
                new_dims.push(
                    check_view(&view_dims[i], &arr_dim, env, ast, tcx).with_context(|| {
                        format!(
                            "failed to type check command: view `{}` dimension {}",
                            id.resolve_id(ast),
                            i
                        )
                    })?,
                );
            }

            let new_ty = tcx.get_array(element_ty, new_dims, ports);

            tcx.value_type_map.insert(*id, new_ty);

            Ok(())
        }
        Command::Split { id, arr_id, dims } => {
            let arr_ty = tcx
                .value_type_map
                .get(arr_id)
                .copied()
                .expect("array should be bound after name resolution");

            let (element_ty, arr_dims, ports) = match &tcx.types[arr_ty] {
                Type::Array {
                    element_type,
                    dims,
                    ports,
                    ..
                } => (*element_type, dims, *ports),
                _ => Err(anyhow!(TypecheckError::UnexpectedType)).with_context(|| {
                    format!(
                        "failed to type check command: split `{}`",
                        id.resolve_id(ast)
                    )
                })?,
            };

            if arr_dims.len() != dims.len() {
                Err(anyhow!(TypecheckError::IncorrectAccessDims)).with_context(|| {
                    format!(
                        "failed to type check command: split `{}`",
                        id.resolve_id(ast)
                    )
                })?;
            }

            let split_dims = dims.iter().zip(arr_dims.iter()).enumerate().try_fold(
                Vec::new(),
                |mut acc, (i, (&split_dim, DimSpec { length, bank }))| {
                    if split_dim == 0 || bank % split_dim != 0 {
                        return Err(anyhow!(TypecheckError::InvalidSplitFactor)).with_context(
                            || {
                                format!(
                                    "failed to type check command: split `{}` dimension {}",
                                    id.resolve_id(ast),
                                    i
                                )
                            },
                        );
                    }
                    acc.push(DimSpec {
                        length: split_dim,
                        bank: split_dim,
                    });
                    acc.push(DimSpec {
                        length: length / split_dim,
                        bank: bank / split_dim,
                    });
                    Ok(acc)
                },
            )?;

            let view_ty = tcx.get_array(element_ty, split_dims, ports);

            tcx.value_type_map.insert(*id, view_ty);

            Ok(())
        }
    }
}

fn check_expr_(
    expr: ExprId,
    env: &mut TypeEnv,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<TypeId> {
    match &ast.exprs[expr] {
        Expr::Placeholder => unreachable!(),
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
            let ty = tcx
                .value_type_map
                .get(id)
                .copied()
                .expect("identifier should be bound after name resolution");
            Ok(ty)
        }
        Expr::BinOp { left, op, right } => {
            let t1 = check_expr(*left, env, ast, tcx)
                .context("failed to type check expression: binary operator LHS")?;
            let t2 = check_expr(*right, env, ast, tcx)
                .context("failed to type check expression: binary operator RHS")?;
            Ok(check_binop(t1, t2, *op, tcx).context("failed to type check binary operation")?)
        }
        Expr::Application { func, args } => {
            let func_ty = env
                .get_func(func)
                .ok_or(TypecheckError::Unbound)
                .with_context(|| {
                    format!(
                        "failed to type check expression: function application `{}`",
                        func.resolve_id(ast)
                    )
                })?;

            // can probably remove this since get_func should guarantee to return a function type
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
                        field.resolve_id(ast)
                    )
                })
        }
        Expr::ArrayAccess { array, indices } => {
            let array_ty = tcx
                .value_type_map
                .get(array)
                .copied()
                .expect("array should be bound after name resolution");
            let (element_ty, dims_len) = match &tcx.types[array_ty] {
                Type::Array {
                    element_type, dims, ..
                } => (*element_type, dims.len()),
                _ => Err(anyhow!(TypecheckError::UnexpectedType)).with_context(|| {
                    format!(
                        "failed to type check expression: array access `{}`",
                        array.resolve_id(ast)
                    )
                })?,
            };

            if indices.len(&ast.expr_lists) != dims_len {
                Err(anyhow!(TypecheckError::IncorrectAccessDims)).with_context(|| {
                    format!(
                        "failed to type check expression: array access `{}` indices",
                        array.resolve_id(ast)
                    )
                })?;
            }

            indices
                .as_slice(&ast.expr_lists)
                .iter()
                .enumerate()
                .try_for_each(|(i, idx)| -> Result<()> {
                    let idx_ty = check_expr(*idx, env, ast, tcx).with_context(|| {
                        format!(
                            "failed to type check expression: array access `{}` index {}",
                            array.resolve_id(ast),
                            i
                        )
                    })?;
                    match &tcx.types[idx_ty] {
                        Type::StaticInt(..) | Type::Bit { .. } | Type::Index { .. } => Ok(()),
                        _ => Err(anyhow!(TypecheckError::UnexpectedType)).with_context(|| {
                            format!(
                                "failed to type check expression: array access `{}` index {}",
                                array.resolve_id(ast),
                                i
                            )
                        }),
                    }
                })?;

            Ok(element_ty)
        }
    }
}

fn check_expr(expr: ExprId, env: &mut TypeEnv, ast: &Ast, tcx: &mut TypeContext) -> Result<TypeId> {
    let ty = check_expr_(expr, env, ast, tcx)?;
    if let Some(prev_ty) = tcx.expr_type_map.get(&expr) {
        assert!(*prev_ty == ty, "Expression type changed during checking");
    }
    tcx.expr_type_map.insert(expr, ty);
    Ok(ty)
}

fn check_binop(tid1: TypeId, tid2: TypeId, op: InfixOp, tcx: &mut TypeContext) -> Result<TypeId> {
    match op {
        InfixOp::Eq | InfixOp::Neq => {
            if let Type::Array { .. } = tcx.types[tid1] {
                return Err(anyhow!(TypecheckError::BinopError)).context(
                    "failed to type check binary operation: equality operator does not support arrays");
            }
            if join_of(tid1, tid2, op, tcx).is_none() {
                return Err(anyhow!(TypecheckError::NoJoin)).context(
                    "failed to type check binary operation: no common supertype for operands",
                );
            }
            Ok(tcx.get_bool())
        }
        InfixOp::And | InfixOp::Or => {
            if tid1 != tcx.get_bool() || tid2 != tcx.get_bool() {
                return Err(anyhow!(TypecheckError::BinopError)).context(
                    "failed to type check binary operation: both operands of bool operation must be bool",
                );
            }
            Ok(tcx.get_bool())
        }
        InfixOp::Mul | InfixOp::Div | InfixOp::Mod | InfixOp::Add | InfixOp::Sub => {
            join_of(tid1, tid2, op, tcx)
                .ok_or(anyhow!(TypecheckError::NoJoin))
                .context("failed to type check binary operation: no common supertype for operands")
        }
        InfixOp::Shl | InfixOp::Shr | InfixOp::Band | InfixOp::Bor | InfixOp::Bxor => {
            match (&tcx.types[tid1], &tcx.types[tid2]) {
                (
                    Type::Bit { length, unsigned },
                    Type::StaticInt(..) | Type::Bit { .. } | Type::Index { .. },
                ) => Ok(tcx.get_bit(*length, *unsigned)),
                (
                    Type::StaticInt(v),
                    Type::StaticInt(..) | Type::Bit { .. } | Type::Index { .. },
                ) => Ok(tcx.get_bit(bits_needed(*v), false)),
                (
                    Type::Index { static_, dynamic },
                    Type::StaticInt(..) | Type::Bit { .. } | Type::Index { .. },
                ) => {
                    let max_val = static_.1 * dynamic.1 - 1;
                    Ok(tcx.get_bit(bits_needed(max_val), false))
                }
                _ => Err(anyhow!(TypecheckError::BinopError)).context(
                    "failed to type check binary operation: invalid operand types for bitwise operation",
                ),
            }
        }
        InfixOp::Lt | InfixOp::Le | InfixOp::Gt | InfixOp::Ge => {
            match (&tcx.types[tid1], &tcx.types[tid2]) {
                (Type::StaticInt(..)|Type::Bit { .. }|Type::Index { .. },
                Type::StaticInt(..)|Type::Bit { .. }|Type::Index { .. }) => Ok(tcx.get_bool()),
                (Type::Float, Type::Float)
                | (Type::Double, Type::Double) => Ok(tcx.get_bool()),
                (
                    Type::Rational(..) | Type::Double | Type::Float | Type::Fixed { .. },
                    Type::Rational(..),
                ) => Ok(tcx.get_bool()),
                (Type::Fixed { .. }, Type::Fixed { .. }) if tid1 == tid2 => Ok(tcx.get_bool()),
                _ => Err(anyhow!(TypecheckError::BinopError)).context(
                    "failed to type check binary operation: invalid operand types for comparison operation",
                ),
            }
        }
    }
}

fn check_view(
    view: &View,
    arr_dim: &DimSpec,
    env: &mut TypeEnv,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<DimSpec> {
    if let Some(shrink) = view.shrink
        && (shrink > arr_dim.bank || !arr_dim.bank.is_multiple_of(shrink))
    {
        return Err(anyhow!(TypecheckError::InvalidShrinkWidth))
            .context("failed to type check view: invalid shrink width");
    }

    let new_bank = view.shrink.unwrap_or(arr_dim.bank);

    let idx = match view.suffix {
        Suffix::Aligned { factor, e } => {
            if new_bank > factor {
                return Err(anyhow!(TypecheckError::InvalidAlignFactor))
                    .context("failed to type check view: invalid align factor");
            }
            if factor % new_bank != 0 {
                return Err(anyhow!(TypecheckError::InvalidAlignFactor))
                    .context("failed to type check view: invalid align factor");
            }
            e
        }
        Suffix::Rotation(idx) => idx,
    };

    let typ = check_expr(idx, env, ast, tcx)?;
    if !matches!(
        &tcx.types[typ],
        Type::StaticInt(..) | Type::Bit { .. } | Type::Index { .. }
    ) {
        return Err(anyhow!(TypecheckError::UnexpectedType))
            .context("failed to type check view: index expression must be integer");
    }

    Ok(DimSpec {
        length: view.prefix.unwrap_or(arr_dim.length),
        bank: new_bank,
    })
}
