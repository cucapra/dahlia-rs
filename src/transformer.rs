use cranelift_entity::EntityList;

use crate::ast::{Ast, Command, CommandId, Expr, ExprId, TypeContext};

pub trait Env {
    fn with_scope<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R;
    fn merge(&mut self, other: Self);
}

pub trait Transformer {
    type E: Env + Clone;

    fn rewrite_expr(
        &mut self,
        expr: ExprId,
        env: &mut Self::E,
        ast: &mut Ast,
        tcx: &mut TypeContext,
    );

    fn rewrite_command(
        &mut self,
        cmd: CommandId,
        env: &mut Self::E,
        ast: &mut Ast,
        tcx: &mut TypeContext,
    );
}

fn rewrite_expr<T: Transformer>(
    t: &mut T,
    expr: ExprId,
    env: &mut T::E,
    ast: &mut Ast,
    tcx: &mut TypeContext,
) {
    enum PendingExpr {
        EntityList(EntityList<ExprId>),
        RecordLiteral(usize),
        BinOp(ExprId, ExprId),
    }

    let pending_expr = match &ast.exprs[expr] {
        Expr::Placeholder => unreachable!(),
        Expr::RationalLiteral(..)
        | Expr::IntLiteral { .. }
        | Expr::BoolLiteral(..)
        | Expr::Id(..) => {
            return;
        }
        Expr::RecordLiteral(fields) => PendingExpr::RecordLiteral(fields.len()),
        Expr::ArrayLiteral(elems) => PendingExpr::EntityList(*elems),
        Expr::BinOp { left, right, .. } => PendingExpr::BinOp(*left, *right),
        Expr::Application { args, .. } => PendingExpr::EntityList(*args),
        Expr::Cast { expr, .. } => {
            t.rewrite_expr(*expr, env, ast, tcx);
            return;
        }
        Expr::RecordAccess { record, .. } => {
            t.rewrite_expr(*record, env, ast, tcx);
            return;
        }
        Expr::ArrayAccess { indices, .. } => PendingExpr::EntityList(*indices),
    };

    match pending_expr {
        PendingExpr::EntityList(exprs) => {
            for i in 0..exprs.len(&ast.expr_lists) {
                t.rewrite_expr(exprs.as_slice(&ast.expr_lists)[i], env, ast, tcx);
            }
        }
        PendingExpr::BinOp(left, right) => {
            t.rewrite_expr(left, env, ast, tcx);
            t.rewrite_expr(right, env, ast, tcx);
        }
        PendingExpr::RecordLiteral(len) => {
            for i in 0..len {
                let field = match &ast.exprs[expr] {
                    Expr::RecordLiteral(fields) => *fields.get_index(i).unwrap().1,
                    _ => unreachable!(),
                };
                t.rewrite_expr(field, env, ast, tcx);
            }
        }
    }
}

fn rewrite_command<T: Transformer>(
    t: &mut T,
    cmd: CommandId,
    env: &mut T::E,
    ast: &mut Ast,
    tcx: &mut TypeContext,
) {
    enum PendingCommand {
        Seq(usize),
        Update(ExprId, ExprId),
        IfElse {
            cond: ExprId,
            then: CommandId,
            else_: CommandId,
        },
        For {
            body: CommandId,
            combine: CommandId,
        },
        While {
            cond: ExprId,
            body: CommandId,
        },
        Block(CommandId),
    }

    let pending_command = match &ast.commands[cmd] {
        Command::Split { .. } | Command::View { .. } | Command::Empty | Command::Decorate(..) => {
            return;
        }
        Command::Par(cmds) | Command::Seq(cmds) => PendingCommand::Seq(cmds.len()),
        Command::Update { lhs, rhs, .. } => PendingCommand::Update(*lhs, *rhs),
        Command::Let { value, .. } => {
            if let Some(value) = value {
                t.rewrite_expr(*value, env, ast, tcx);
            }
            return;
        }
        Command::Expr(expr) => {
            t.rewrite_expr(*expr, env, ast, tcx);
            return;
        }
        Command::Return(expr) => {
            t.rewrite_expr(*expr, env, ast, tcx);
            return;
        }
        Command::IfElse { cond, then, else_ } => PendingCommand::IfElse {
            cond: *cond,
            then: *then,
            else_: *else_,
        },
        Command::For { body, combine, .. } => PendingCommand::For {
            body: *body,
            combine: *combine,
        },
        Command::While { cond, body, .. } => PendingCommand::While {
            cond: *cond,
            body: *body,
        },
        Command::Block(body) => PendingCommand::Block(*body),
    };

    match pending_command {
        PendingCommand::Seq(len) => {
            for i in 0..len {
                let cmd = match &ast.commands[cmd] {
                    Command::Par(cmds) | Command::Seq(cmds) => cmds[i],
                    _ => unreachable!(),
                };
                rewrite_command(t, cmd, env, ast, tcx);
            }
        }
        PendingCommand::Update(lhs, rhs) => {
            t.rewrite_expr(lhs, env, ast, tcx);
            t.rewrite_expr(rhs, env, ast, tcx);
        }
        PendingCommand::IfElse { cond, then, else_ } => {
            t.rewrite_expr(cond, env, ast, tcx);
            let mut new_env = env.clone();
            env.with_scope(|env| t.rewrite_command(then, env, ast, tcx));
            new_env.with_scope(|env| t.rewrite_command(else_, env, ast, tcx));
            env.merge(new_env);
        }
        PendingCommand::For { body, combine } => {
            env.with_scope(|env| t.rewrite_command(body, env, ast, tcx));
            t.rewrite_command(combine, env, ast, tcx);
        }
        PendingCommand::While { cond, body } => {
            t.rewrite_expr(cond, env, ast, tcx);
            env.with_scope(|env| t.rewrite_command(body, env, ast, tcx));
        }
        PendingCommand::Block(body) => {
            env.with_scope(|env| t.rewrite_command(body, env, ast, tcx));
        }
    }
}
