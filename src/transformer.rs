use anyhow::{Context, Result};

use crate::ast::{
    Ast, Command, CommandId, Decl, Def, Expr, ExprId, IdResolve, Program, TypeContext,
};

pub trait Env {
    fn new() -> Self;
    fn with_scope<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R;
    fn merge(&mut self, other: Self);
}

pub trait Transformer {
    type E: Env + Clone;

    const NAME: &'static str;

    // when not handling a command or expression, the implemented trait function should call the corresponding top-level rewrite function
    fn rewrite_expr(
        expr: ExprId,
        env: &mut Self::E,
        ast: &mut Ast,
        tcx: &mut TypeContext,
    ) -> Result<()>;

    fn rewrite_lval(
        expr: ExprId,
        env: &mut Self::E,
        ast: &mut Ast,
        tcx: &mut TypeContext,
    ) -> Result<()> {
        Self::rewrite_expr(expr, env, ast, tcx)
    }

    fn rewrite_decl(
        _decl: &mut Decl,
        _env: &mut Self::E,
        _ast: &mut Ast,
        _tcx: &mut TypeContext,
    ) -> Result<()> {
        Ok(())
    }

    fn rewrite_command(
        cmd: CommandId,
        env: &mut Self::E,
        ast: &mut Ast,
        tcx: &mut TypeContext,
    ) -> Result<()>;
}

pub fn rewrite_expr<T: Transformer>(
    expr: ExprId,
    env: &mut T::E,
    ast: &mut Ast,
    tcx: &mut TypeContext,
) -> Result<()> {
    let extracted_expr = std::mem::replace(&mut ast.exprs[expr], Expr::Placeholder);

    match &extracted_expr {
        Expr::Placeholder => unreachable!(),
        Expr::RationalLiteral(..)
        | Expr::IntLiteral { .. }
        | Expr::BoolLiteral(..)
        | Expr::Id(..) => {}
        Expr::RecordLiteral(fields) => {
            fields
                .values()
                .try_for_each(|field| T::rewrite_expr(*field, env, ast, tcx))
                .with_context(|| format!("{}: failed to rewrite record literal", T::NAME))?;
        }
        Expr::ArrayLiteral(elems) => {
            for i in 0..elems.len(&ast.expr_lists) {
                T::rewrite_expr(elems.as_slice(&ast.expr_lists)[i], env, ast, tcx)
                    .with_context(|| format!("{}: failed to rewrite array literal", T::NAME))?;
            }
        }
        Expr::BinOp { left, right, .. } => {
            T::rewrite_expr(*left, env, ast, tcx)
                .with_context(|| format!("{}: failed to rewrite binary operation LHS", T::NAME))?;
            T::rewrite_expr(*right, env, ast, tcx)
                .with_context(|| format!("{}: failed to rewrite binary operation RHS", T::NAME))?;
        }
        Expr::Application { args, .. } => {
            for i in 0..args.len(&ast.expr_lists) {
                T::rewrite_expr(args.as_slice(&ast.expr_lists)[i], env, ast, tcx).with_context(
                    || format!("{}: failed to rewrite application argument", T::NAME),
                )?;
            }
        }
        Expr::Cast { expr, .. } => {
            T::rewrite_expr(*expr, env, ast, tcx)
                .with_context(|| format!("{}: failed to rewrite cast expression", T::NAME))?;
        }
        Expr::RecordAccess { record, .. } => {
            T::rewrite_expr(*record, env, ast, tcx).with_context(|| {
                format!("{}: failed to rewrite record access expression", T::NAME)
            })?;
        }
        Expr::ArrayAccess { indices, .. } => {
            for i in 0..indices.len(&ast.expr_lists) {
                T::rewrite_expr(indices.as_slice(&ast.expr_lists)[i], env, ast, tcx).with_context(
                    || format!("{}: failed to rewrite array access expression", T::NAME),
                )?;
            }
        }
    }

    ast.exprs[expr] = extracted_expr;
    Ok(())
}

pub fn rewrite_command<T: Transformer>(
    cmd: CommandId,
    env: &mut T::E,
    ast: &mut Ast,
    tcx: &mut TypeContext,
) -> Result<()> {
    let extracted_cmd = std::mem::replace(&mut ast.commands[cmd], Command::Empty);

    match &extracted_cmd {
        Command::Empty | Command::Split { .. } | Command::View { .. } | Command::Decorate(..) => {}
        Command::Par(cmds) => {
            cmds.iter()
                .try_for_each(|cmd| T::rewrite_command(*cmd, env, ast, tcx))
                .with_context(|| format!("{}: failed to rewrite Par command", T::NAME))?;
        }
        Command::Seq(cmds) => {
            cmds.iter()
                .try_for_each(|cmd| T::rewrite_command(*cmd, env, ast, tcx))
                .with_context(|| format!("{}: failed to rewrite Seq command", T::NAME))?;
        }
        Command::Update { lhs, rhs, .. } => {
            T::rewrite_lval(*lhs, env, ast, tcx)
                .with_context(|| format!("{}: failed to rewrite update command LHS", T::NAME))?;
            T::rewrite_expr(*rhs, env, ast, tcx)
                .with_context(|| format!("{}: failed to rewrite update command RHS", T::NAME))?;
        }
        Command::Let { value, .. } => {
            if let Some(value) = value {
                T::rewrite_expr(*value, env, ast, tcx).with_context(|| {
                    format!("{}: failed to rewrite let binding initializer", T::NAME)
                })?;
            }
        }
        Command::Expr(expr) => {
            T::rewrite_expr(*expr, env, ast, tcx)
                .with_context(|| format!("{}: failed to rewrite expression command", T::NAME))?;
        }
        Command::Return(expr) => {
            T::rewrite_expr(*expr, env, ast, tcx)
                .with_context(|| format!("{}: failed to rewrite return command", T::NAME))?;
        }
        Command::IfElse { cond, then, else_ } => {
            T::rewrite_expr(*cond, env, ast, tcx)
                .with_context(|| format!("{}: failed to rewrite if condition", T::NAME))?;
            let mut new_env = env.clone();
            env.with_scope(|env| T::rewrite_command(*then, env, ast, tcx))
                .with_context(|| format!("{}: failed to rewrite then branch", T::NAME))?;
            new_env
                .with_scope(|env| T::rewrite_command(*else_, env, ast, tcx))
                .with_context(|| format!("{}: failed to rewrite else branch", T::NAME))?;
            env.merge(new_env);
        }
        Command::For { body, combine, .. } => {
            env.with_scope(|env| -> Result<()> {
                T::rewrite_command(*body, env, ast, tcx)
                    .with_context(|| format!("{}: failed to rewrite for body", T::NAME))?;
                T::rewrite_command(*combine, env, ast, tcx)
                    .with_context(|| format!("{}: failed to rewrite for combine", T::NAME))?;
                Ok(())
            })?;
        }
        Command::While { cond, body, .. } => {
            T::rewrite_expr(*cond, env, ast, tcx)
                .with_context(|| format!("{}: failed to rewrite while condition", T::NAME))?;
            env.with_scope(|env| {
                T::rewrite_command(*body, env, ast, tcx)
                    .with_context(|| format!("{}: failed to rewrite while body", T::NAME))
            })?;
        }
        Command::Block(body) => {
            env.with_scope(|env| {
                T::rewrite_command(*body, env, ast, tcx)
                    .with_context(|| format!("{}: failed to rewrite block", T::NAME))
            })?;
        }
    }

    ast.commands[cmd] = extracted_cmd;
    Ok(())
}

fn rewrite_def<T: Transformer>(
    def: &mut Def,
    env: &mut T::E,
    ast: &mut Ast,
    tcx: &mut TypeContext,
) -> Result<()> {
    if let Def::Func { sig, body } = def {
        env.with_scope(|env| {
            sig.args
                .iter_mut()
                .try_for_each(|decl| T::rewrite_decl(decl, env, ast, tcx))
                .with_context(|| {
                    format!(
                        "{}: failed to rewrite function arguments for `{}`",
                        T::NAME,
                        sig.name.resolve_id(ast)
                    )
                })?;

            T::rewrite_command(*body, env, ast, tcx).with_context(|| {
                format!(
                    "{}: failed to rewrite function body for `{}`",
                    T::NAME,
                    sig.name.resolve_id(ast)
                )
            })
        })?;
    }

    Ok(())
}

pub fn rewrite_program<T: Transformer>(
    prog: &mut Program,
    ctx: &mut crate::ast::Context,
) -> Result<()> {
    let mut env = T::E::new();
    let ast = &mut ctx.ast;
    let tcx = &mut ctx.tcx;

    prog.defs
        .iter_mut()
        .try_for_each(|def| rewrite_def::<T>(def, &mut env, ast, tcx))
        .with_context(|| format!("{}: failed to rewrite program definitions", T::NAME))?;

    prog.decls
        .iter_mut()
        .try_for_each(|decl| T::rewrite_decl(decl, &mut env, ast, tcx))
        .with_context(|| format!("{}: failed to rewrite program declarations", T::NAME))?;

    T::rewrite_command(prog.cmd, &mut env, ast, tcx)
        .with_context(|| format!("{}: failed to rewrite program command", T::NAME))?;

    Ok(())
}
