use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use calyx_ir::Attributes;
use calyx_ir::BoolAttr;
use calyx_ir::Component;
use calyx_ir::PortDef;
use calyx_ir::{Builder, Cell, Control, Port, RRC};
use indexmap::IndexMap;

use crate::ast::DimSpec;
use crate::ast::IdResolve;
use crate::ast::Type;
use crate::ast::TypeId;
use crate::ast::{Ast, CommandId, Decl, Def, ExprId, Program, TypeContext, ValueId};
use crate::utils::bits_needed;

type Guard = calyx_ir::Guard<calyx_ir::Nothing>;
type Assignment = calyx_ir::Assignment<calyx_ir::Nothing>;

struct ExprEmitOutput {
    output: RRC<Port>,
    done: Option<RRC<Port>>,
    assignments: Vec<Assignment>,
}

#[derive(Debug)]
enum Structure {
    Cell(RRC<Cell>),
    Port(RRC<Port>),
}

type Env = IndexMap<ValueId, Structure>;

fn bits_for_type(ty: TypeId, tcx: &TypeContext) -> (u64, Option<u64>) {
    match &tcx.types[ty] {
        Type::Bit { length, .. } => (*length as u64, None),
        Type::Fixed {
            length_total,
            length_int,
            ..
        } => (*length_total as u64, Some(*length_int as u64)),
        Type::Bool => (1, None),
        Type::Void => (0, None),
        _ => unimplemented!("{:?}", tcx.types[ty]),
    }
}

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
    Ok(Control::empty())
    // todo!()
}

fn emit_array_decl(
    element_type: &TypeId,
    dims: &Vec<DimSpec>,
    ports: &usize,
    fn_arg: bool,
    builder: &mut Builder,
    tcx: &TypeContext,
) -> Result<Structure> {
    if *ports != 1 {
        bail!("Multi-port memories not supported");
    }
    if !dims.iter().all(|dim| dim.bank == 1) {
        bail!("Bankd memories should be lowered");
    }
    if dims.len() > 4 {
        bail!("Memories with more than 4 dimensions not supported");
    }

    let (width, _) = bits_for_type(*element_type, tcx);
    let mut params = vec![width];
    params.extend(dims.iter().map(|dim| dim.length as u64));
    // TODO: better error handling around casts perhaps
    params.extend(dims.iter().map(|dim| bits_needed(dim.length as i64) as u64));

    let cell = builder.add_primitive("mem", format!("seq_mem_d{}", dims.len()), &params);

    if !fn_arg {
        cell.borrow_mut().add_attribute(BoolAttr::External, 1);
    }
    cell.borrow_mut().set_reference(fn_arg);

    Ok(Structure::Cell(cell))
}

// top-level decl becomes either an external memory cell or a register
fn emit_decl(decl: &Decl, env: &mut Env, builder: &mut Builder, tcx: &TypeContext) -> Result<()> {
    let tid = tcx.value_type_map[&decl.id];
    let structure = match &tcx.types[tid] {
        Type::Array {
            element_type,
            dims,
            ports,
        } => emit_array_decl(element_type, dims, ports, false, builder, tcx)?,
        _ => {
            let (width, _) = bits_for_type(tid, tcx);
            Structure::Cell(builder.add_primitive("reg", "std_reg", &[width]))
        }
    };
    env.insert(decl.id, structure);
    Ok(())
}

fn emit_def(
    def: &Def,
    calyx_ast: &mut calyx_ir::Context,
    dahlia_ast: &Ast,
    tcx: &TypeContext,
) -> Result<()> {
    match def {
        // TODO: refactor bail! macros into proper error types
        Def::Record { .. } => bail!("record definitions not supported yet"),
        Def::Func { sig, body } => {
            let mut env = Env::new();

            let mut ports: Vec<_> = sig
                .args
                .iter()
                .filter(|arg| {
                    !matches!(&tcx.types[tcx.value_type_map[&arg.id]], Type::Array { .. })
                })
                .map(|arg| {
                    let (width, _) = bits_for_type(tcx.value_type_map[&arg.id], tcx);
                    PortDef::new(
                        arg.id.get_name(),
                        width,
                        calyx_ir::Direction::Input,
                        Attributes::default(),
                    )
                })
                .collect();

            // add output port if return type is not void
            if !matches!(&tcx.types[sig.ret_ty], Type::Void) {
                let (width, _) = bits_for_type(sig.ret_ty, tcx);

                let mut attrs = Attributes::default();
                attrs.insert(BoolAttr::Stable, 1);

                ports.push(PortDef::new(
                    "out",
                    width,
                    calyx_ir::Direction::Output,
                    attrs,
                ));
            }

            let mut component = Component::new(sig.name.get_name(), ports, true, false, None);

            for arg in &sig.args {
                if !matches!(&tcx.types[tcx.value_type_map[&arg.id]], Type::Array { .. }) {
                    env.insert(
                        arg.id,
                        Structure::Port(component.signature.borrow().get(arg.id.get_name())),
                    );
                }
            }

            let mut builder = Builder::new(&mut component, &calyx_ast.lib);
            for arg in &sig.args {
                if let Type::Array {
                    element_type,
                    dims,
                    ports,
                } = &tcx.types[tcx.value_type_map[&arg.id]]
                {
                    let structure =
                        emit_array_decl(element_type, dims, ports, true, &mut builder, tcx)?;
                    env.insert(arg.id, structure);
                }
            }

            *component.control.borrow_mut() =
                emit_command(*body, &mut env, &mut builder, dahlia_ast, tcx).with_context(
                    || {
                        format!(
                            "failed to emit function {} body",
                            sig.name.resolve_id(&dahlia_ast)
                        )
                    },
                )?;

            calyx_ast.components.push(component);
        }
    }
    Ok(())
}

pub fn emit_calyx(
    program: &Program,
    calyx_ast: &mut calyx_ir::Context,
    ctx: &crate::ast::Context,
) -> Result<()> {
    program
        .includes
        .iter()
        .flat_map(|include| &include.defs)
        .chain(&program.defs)
        .try_for_each(|def| {
            emit_def(def, calyx_ast, &ctx.ast, &ctx.tcx).with_context(|| {
                format!(
                    "failed to emit definition {}",
                    match def {
                        Def::Record { name, .. } => name.resolve_id(&ctx.ast),
                        Def::Func { sig, .. } => sig.name.resolve_id(&ctx.ast),
                    }
                )
            })
        })?;

    // TODO: handle Calyx imports

    let mut env = Env::new();
    let mut main_component = Component::new("main", vec![], true, false, None);
    let mut builder = Builder::new(&mut main_component, &calyx_ast.lib);

    program.decls.iter().try_for_each(|decl| {
        emit_decl(decl, &mut env, &mut builder, &ctx.tcx).with_context(|| {
            format!(
                "failed to emit declaration {}",
                decl.id.resolve_id(&ctx.ast)
            )
        })
    })?;

    *main_component.control.borrow_mut() =
        emit_command(program.cmd, &mut env, &mut builder, &ctx.ast, &ctx.tcx)
            .context("failed to emit the main command")?;

    calyx_ast.components.push(main_component);
    Ok(())
}
