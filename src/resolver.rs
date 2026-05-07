use anyhow::{Context, Result, anyhow};

use crate::{
    ast::{Ast, Command, CommandId, Decl, Def, Expr, ExprId, IdResolve, Program, Suffix},
    errors::ResolveError,
    resolver_env::ResolverEnv,
};

pub fn resolve_names(program: &Program, context: &mut crate::ast::Context) -> Result<()> {
    let ast = &mut context.ast;
    let mut env = ResolverEnv::new();

    program
        .includes
        .iter()
        .flat_map(|include| &include.defs)
        .chain(&program.defs)
        .try_for_each(|def| {
            resolve_def(def, &mut env, ast).with_context(|| {
                format!(
                    "failed to resolve definition `{}`",
                    match def {
                        Def::Record { name, .. } => name.resolve_id(ast),
                        Def::Func { sig, .. } => sig.name.resolve_id(ast),
                    }
                )
            })
        })?;

    for decl in &program.decls {
        resolve_decl(decl, &mut env, ast).with_context(|| {
            format!(
                "failed to resolve declaration `{}`",
                decl.id.resolve_id(ast)
            )
        })?;
    }

    resolve_command(program.cmd, &mut env, ast).context("failed to resolve the main command")?;

    Ok(())
}

fn resolve_decl(decl: &Decl, env: &mut ResolverEnv, ast: &mut Ast) -> Result<()> {
    env.add(decl.id, ast)
        .with_context(|| format!("`{}` already bound", decl.id.resolve_id(ast)))?;
    Ok(())
}

fn resolve_def(def: &Def, env: &mut ResolverEnv, ast: &mut Ast) -> Result<()> {
    // only function definitions need name resolution
    if let Def::Func { sig, body } = def {
        env.with_scope(|env| -> Result<()> {
            for decl in &sig.args {
                resolve_decl(decl, env, ast).with_context(|| {
                    format!(
                        "failed to resolve argument `{}` in function `{}`",
                        decl.id.resolve_id(ast),
                        sig.name.resolve_id(ast)
                    )
                })?;
            }

            resolve_command(*body, env, ast).with_context(|| {
                format!(
                    "failed to resolve function `{}` body",
                    sig.name.resolve_id(ast)
                )
            })?;

            Ok(())
        })?;
    }
    Ok(())
}

fn resolve_command(cmd: CommandId, env: &mut ResolverEnv, ast: &mut Ast) -> Result<()> {
    let mut extracted_cmd = std::mem::replace(&mut ast.commands[cmd], Command::Empty);
    match &extracted_cmd {
        Command::Block(cmd) => env.with_scope(|env| {
            resolve_command(*cmd, env, ast).context("failed to resolve block command")
        })?,
        Command::Seq(cmds) => {
            cmds.iter()
                .try_for_each(|cmd| resolve_command(*cmd, env, ast))
                .context("failed to resolve Seq command")?;
        }
        Command::Par(cmds) => {
            cmds.iter()
                .try_for_each(|cmd| resolve_command(*cmd, env, ast))
                .context("failed to resolve Par command")?;
        }
        Command::IfElse { cond, then, else_ } => {
            resolve_expr(*cond, env, ast).context("failed to resolve if condition")?;
            env.with_scope(|env| {
                resolve_command(*then, env, ast).context("failed to resolve then branch")
            })?;
            env.with_scope(|env| {
                resolve_command(*else_, env, ast).context("failed to resolve else branch")
            })?;
        }
        Command::For {
            range,
            body,
            combine,
            ..
        } => env.with_scope(|env| -> Result<()> {
            env.add(range.iter, ast)
                .with_context(|| format!("`{}` already bound", range.iter.resolve_id(ast)))?;
            resolve_command(*body, env, ast).context("failed to resolve for loop body")?;
            resolve_command(*combine, env, ast).context("failed to resolve for loop combine")?;
            Ok(())
        })?,
        Command::While { cond, body, .. } => {
            resolve_expr(*cond, env, ast).context("failed to resolve while loop condition")?;
            env.with_scope(|env| {
                resolve_command(*body, env, ast).context("failed to resolve while loop body")
            })?;
        }
        Command::Update { lhs, rhs, .. } => {
            resolve_expr(*lhs, env, ast).context("failed to resolve update lhs")?;
            resolve_expr(*rhs, env, ast).context("failed to resolve update rhs")?;
        }
        Command::Let { id, value, .. } => {
            if let Some(value) = value {
                resolve_expr(*value, env, ast).context("failed to resolve let initializer")?;
            }
            env.add(*id, ast)
                .with_context(|| format!("`{}` already bound", id.resolve_id(ast)))?;
        }
        Command::Expr(expr) => {
            resolve_expr(*expr, env, ast).context("failed to resolve expression command")?;
        }
        Command::Return(expr) => {
            resolve_expr(*expr, env, ast).context("failed to resolve return expression")?;
        }
        Command::View { id, arr_id, dims } => {
            dims.iter().try_for_each(|dim| match &dim.suffix {
                Suffix::Rotation(expr) | Suffix::Aligned { e: expr, .. } => {
                    resolve_expr(*expr, env, ast).with_context(|| {
                        format!("failed to resolve view `{}` dimension", id.resolve_id(ast))
                    })
                }
            })?;
            if env.get(arr_id, ast).is_none() {
                return Err(anyhow!(ResolveError::Unbound)).with_context(|| {
                    format!(
                        "failed to resolve view `{}`: array `{}` is unbound",
                        id.resolve_id(ast),
                        arr_id.resolve_id(ast)
                    )
                });
            }
            env.add(*id, ast)
                .with_context(|| format!("`{}` already bound", id.resolve_id(ast)))?;
        }
        Command::Split { id, arr_id, .. } => {
            if env.get(arr_id, ast).is_none() {
                return Err(anyhow!(ResolveError::Unbound)).with_context(|| {
                    format!(
                        "failed to resolve split `{}`: array `{}` is unbound",
                        id.resolve_id(ast),
                        arr_id.resolve_id(ast)
                    )
                });
            }
            env.add(*id, ast)
                .with_context(|| format!("`{}` already bound", id.resolve_id(ast)))?;
        }
        _ => {}
    }

    if let Command::Split { arr_id, .. } | Command::View { arr_id, .. } = &mut extracted_cmd {
        *arr_id = env.get(arr_id, ast).expect("Array ID should be bound"); // since we already checked in the above match
    }

    ast.commands[cmd] = extracted_cmd;
    Ok(())
}

fn resolve_expr(expr: ExprId, env: &mut ResolverEnv, ast: &mut Ast) -> Result<()> {
    let mut extracted_expr = std::mem::replace(&mut ast.exprs[expr], Expr::Placeholder);

    match &extracted_expr {
        Expr::Placeholder => unreachable!(),
        Expr::Cast { expr, .. } => {
            resolve_expr(*expr, env, ast).context("failed to resolve cast expression")?;
        }
        Expr::Id(id) => {
            if env.get(id, ast).is_none() {
                return Err(anyhow!(ResolveError::Unbound)).with_context(|| {
                    format!("failed to resolve identifier `{}`", id.resolve_id(ast))
                });
            }
        }
        Expr::BinOp { left, right, .. } => {
            resolve_expr(*left, env, ast)
                .context("failed to resolve left operand of binary operation")?;
            resolve_expr(*right, env, ast)
                .context("failed to resolve right operand of binary operation")?;
        }
        Expr::Application { args, .. } => {
            for i in 0..args.len(&ast.expr_lists) {
                resolve_expr(args.as_slice(&ast.expr_lists)[i], env, ast)
                    .context("failed to resolve application argument")?;
            }
        }
        Expr::RecordAccess { record, .. } => {
            resolve_expr(*record, env, ast).context("failed to resolve record access")?;
        }
        Expr::ArrayAccess { indices, .. } => {
            for i in 0..indices.len(&ast.expr_lists) {
                resolve_expr(indices.as_slice(&ast.expr_lists)[i], env, ast)
                    .context("failed to resolve array access index")?;
            }
        }
        Expr::ArrayLiteral(elems) => {
            for i in 0..elems.len(&ast.expr_lists) {
                resolve_expr(elems.as_slice(&ast.expr_lists)[i], env, ast)
                    .context("failed to resolve array literal element")?;
            }
        }
        Expr::RecordLiteral(fields) => {
            for i in 0..fields.len() {
                let field = match &extracted_expr {
                    Expr::RecordLiteral(fields) => *fields.get_index(i).unwrap().1,
                    _ => unreachable!(),
                };
                resolve_expr(field, env, ast).context("failed to resolve record literal field")?;
            }
        }
        Expr::RationalLiteral(..) | Expr::IntLiteral { .. } | Expr::BoolLiteral(..) => {}
    }

    if let Expr::Id(id) | Expr::ArrayAccess { array: id, .. } = &mut extracted_expr {
        *id = env.get(id, ast).expect("ID should be bound"); // since we already checked in the above match
    }

    ast.exprs[expr] = extracted_expr;
    Ok(())
}
