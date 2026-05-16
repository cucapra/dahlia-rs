use anyhow::Context;
use anyhow::Result;
use calyx_ir::{Builder, Cell, Control, Port, RRC};
use indexmap::IndexMap;

use crate::ast::{Ast, CommandId, Decl, Def, ExprId, Program, TypeContext, ValueId};

type Assignment = calyx_ir::Assignment<calyx_ir::Nothing>;

struct ExprEmitOutput {
    output: RRC<Port>,
    done: Option<RRC<Port>>,
    assignments: Vec<Assignment>,
}

enum Structure {
    Cell(RRC<Cell>),
    Port(RRC<Port>),
}

type Env = IndexMap<ValueId, Structure>;

fn emit_invoke(
    app: ExprId,
    env: &mut Env,
    builder: &mut Builder,
    ast: &Ast,
    tcx: &TypeContext,
) -> Result<(Cell, Vec<Assignment>, Control)> {
    todo!()
}

fn emit_lvalue(
    expr: ExprId,
    env: &mut Env,
    builder: &mut Builder,
    ast: &Ast,
    tcx: &TypeContext,
) -> Result<ExprEmitOutput> {
    todo!()
}

fn emit_expr(
    expr: ExprId,
    env: &mut Env,
    builder: &mut Builder,
    ast: &Ast,
    tcx: &TypeContext,
) -> Result<ExprEmitOutput> {
    todo!()
}

fn emit_command(
    cmd: CommandId,
    env: &mut Env,
    builder: &mut Builder,
    ast: &Ast,
    tcx: &TypeContext,
) -> Result<Control> {
    todo!()
}

fn emit_decl(
    decl: &Decl,
    env: &mut Env,
    builder: &mut Builder,
    ast: &Ast,
    tcx: &TypeContext,
) -> Result<()> {
    todo!()
}

fn emit_def(
    def: &Def,
    env: &mut Env,
    calyx_ast: &mut calyx_ir::Context,
    dahlia_ast: &Ast,
    tcx: &TypeContext,
) -> Result<()> {
    todo!()
}

pub fn emit_calyx(
    program: &Program,
    calyx_ast: &mut calyx_ir::Context,
    ctx: &crate::ast::Context,
) -> Result<()> {
    todo!()
}
