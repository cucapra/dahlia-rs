use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use calyx_ir::Attributes;
use calyx_ir::BoolAttr;
use calyx_ir::Component;
use calyx_ir::Id;
use calyx_ir::Invoke;
use calyx_ir::PortDef;
use calyx_ir::{Builder, Cell, Control, Port, RRC};
use cranelift_entity::EntityList;
use indexmap::IndexMap;

use crate::ast::Command;
use crate::ast::DimSpec;
use crate::ast::Expr;
use crate::ast::FuncId;
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

#[derive(Default)]
struct Env {
    value_map: IndexMap<ValueId, Structure>,
    func_port_map: IndexMap<FuncId, Vec<PortDef<u64>>>,
    // refcell / port Id, is refcell?
    func_arg_map: IndexMap<FuncId, Vec<(Id, bool)>>,
}

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
    func: FuncId,
    args: EntityList<ExprId>,
    env: &mut Env,
    builder: &mut Builder,
    ast: &Ast,
    tcx: &TypeContext,
) -> Result<(RRC<Cell>, Vec<Assignment>, Control)> {
    let comp = builder.add_component("invoke", &func.get_name(), env.func_port_map[&func].clone());

    let mut assignments = Vec::new();
    let mut inputs = Vec::new();
    let mut ref_cells = Vec::new();

    for (i, arg) in args.as_slice(&ast.expr_lists).iter().enumerate() {
        let emit_output = emit_expr(*arg, env, builder, ast, tcx).with_context(|| {
            format!(
                "failed to emit arg for function application {}",
                func.resolve_id(ast)
            )
        })?;
        let (id, is_refcell) = &env.func_arg_map[&func][i];

        assignments.extend(emit_output.assignments.into_iter());

        if *is_refcell {
            ref_cells.push((id.clone(), emit_output.output.borrow_mut().cell_parent()));
        } else {
            inputs.push((id.clone(), emit_output.output));
        }
    }

    let invoke_control = Control::Invoke(Invoke {
        attributes: Attributes::default(),
        comp: comp.clone(),
        inputs,
        outputs: vec![],
        comb_group: None,
        ref_cells,
    });

    Ok((comp, assignments, invoke_control))
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
    match &ast.commands[cmd] {
        Command::Block(cmd) => {
            Ok(emit_command(*cmd, env, builder, ast, tcx)
                .context("failed to emit block command")?)
        }
        Command::Par(cmds) => Ok(Control::par(
            cmds.iter()
                .map(|cmd| emit_command(*cmd, env, builder, ast, tcx))
                .collect::<Result<Vec<_>>>()
                .context("failed to emit parallel command")?,
        )),
        Command::Seq(cmds) => Ok(Control::seq(
            cmds.iter()
                .map(|cmd| emit_command(*cmd, env, builder, ast, tcx))
                .collect::<Result<Vec<_>>>()
                .context("failed to emit seq command")?,
        )),
        Command::Let { id, value, .. } => {
            match (
                &tcx.types[tcx.value_type_map[id]],
                value.map(|value| &ast.exprs[value]),
            ) {
                (
                    Type::Array {
                        element_type,
                        dims,
                        ports,
                    },
                    None,
                ) => {
                    let structure = emit_array_decl(element_type, dims, ports, false, builder, tcx)
                        .with_context(|| {
                            format!("failed to emit let binding `{}`", id.resolve_id(ast))
                        })?;
                    env.value_map.insert(*id, structure);
                    Ok(Control::empty())
                }
                (Type::Array { .. }, Some(_)) => {
                    bail!("Calyx backend does not support array initializers");
                }
                (_, Some(Expr::Application { func, args })) => {
                    let (cell, assigns, invoke_control) =
                        emit_invoke(*func, *args, env, builder, ast, tcx).with_context(|| {
                            format!("failed to emit let binding `{}`", id.resolve_id(ast))
                        })?;

                    let (width, _) = bits_for_type(tcx.value_type_map[id], tcx);

                    let reg = builder.add_primitive("reg", "std_reg", &[width]);

                    let group = builder.add_group("let_invoke");

                    let reg_in = reg.borrow().get("in");
                    let invoke_out = cell.borrow().get("out");

                    let reg_write_en = reg.borrow().get("write_en");
                    let const_1 = builder.add_constant(1, 1).borrow().get("out");

                    let group_done = group.borrow().get("done");
                    let reg_done = reg.borrow().get("done");

                    group
                        .borrow_mut()
                        .assignments
                        .push(builder.build_assignment(reg_in, invoke_out, Guard::True));
                    group
                        .borrow_mut()
                        .assignments
                        .push(builder.build_assignment(reg_write_en, const_1, Guard::True));
                    group
                        .borrow_mut()
                        .assignments
                        .push(builder.build_assignment(group_done, reg_done, Guard::True));

                    let control = Control::seq(vec![invoke_control, Control::enable(group)]);

                    env.value_map.insert(*id, Structure::Cell(reg));

                    builder.add_continuous_assignments(assigns);

                    Ok(control)
                }
                (Type::Fixed { length_total, .. }, Some(_)) => {
                    let reg = builder.add_primitive("reg", "std_reg", &[*length_total as u64]);

                    let out =
                        emit_expr(value.unwrap(), env, builder, ast, tcx).with_context(|| {
                            format!(
                                "failed to emit initializer for let binding `{}`",
                                id.resolve_id(ast)
                            )
                        })?;

                    let group = builder.add_group("let_fixed");

                    let reg_in = reg.borrow().get("in");
                    let out_out = out.output;

                    let reg_write_en = reg.borrow().get("write_en");
                    let done = out
                        .done
                        .unwrap_or_else(|| builder.add_constant(1, 1).borrow().get("out"));

                    let group_done = group.borrow().get("done");
                    let reg_done = reg.borrow().get("done");

                    group
                        .borrow_mut()
                        .assignments
                        .push(builder.build_assignment(reg_in, out_out, Guard::True));
                    group
                        .borrow_mut()
                        .assignments
                        .push(builder.build_assignment(reg_write_en, done, Guard::True));
                    group
                        .borrow_mut()
                        .assignments
                        .push(builder.build_assignment(group_done, reg_done, Guard::True));

                    builder.add_continuous_assignments(out.assignments);

                    env.value_map.insert(*id, Structure::Cell(reg));

                    Ok(Control::enable(group))
                }
                (_, Some(_)) => {
                    let (width, _) = bits_for_type(tcx.value_type_map[id], tcx);
                    let reg = builder.add_primitive("reg", "std_reg", &[width]);

                    let out =
                        emit_expr(value.unwrap(), env, builder, ast, tcx).with_context(|| {
                            format!(
                                "failed to emit initializer for let binding `{}`",
                                id.resolve_id(ast)
                            )
                        })?;

                    let group = builder.add_group("let_init");

                    let reg_in = reg.borrow().get("in");
                    let out_out = out.output;

                    let reg_write_en = reg.borrow().get("write_en");
                    let done = out
                        .done
                        .unwrap_or_else(|| builder.add_constant(1, 1).borrow().get("out"));

                    let group_done = group.borrow().get("done");
                    let reg_done = reg.borrow().get("done");

                    group
                        .borrow_mut()
                        .assignments
                        .push(builder.build_assignment(reg_in, out_out, Guard::True));
                    group
                        .borrow_mut()
                        .assignments
                        .push(builder.build_assignment(reg_write_en, done, Guard::True));
                    group
                        .borrow_mut()
                        .assignments
                        .push(builder.build_assignment(group_done, reg_done, Guard::True));

                    builder.add_continuous_assignments(out.assignments);

                    env.value_map.insert(*id, Structure::Cell(reg));

                    Ok(Control::enable(group))
                }
                (_, None) => {
                    let (width, _) = bits_for_type(tcx.value_type_map[id], tcx);
                    let reg = builder.add_primitive("reg", "std_reg", &[width]);
                    env.value_map.insert(*id, Structure::Cell(reg));
                    Ok(Control::empty())
                }
            }
        }
        _ => todo!(),
    }
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
    env.value_map.insert(decl.id, structure);
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
            let mut env = Env::default();

            let mut arg_ids = vec![(Id::default(), false); sig.args.len()];
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

            // could avoid cloning by inserting indices into ast.components into func_map instead?
            env.func_port_map.insert(sig.name, ports.clone());

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

            for (i, arg) in sig.args.iter().enumerate() {
                if !matches!(&tcx.types[tcx.value_type_map[&arg.id]], Type::Array { .. }) {
                    arg_ids[i] = (arg.id.get_name().into(), false);
                    env.value_map.insert(
                        arg.id,
                        Structure::Port(component.signature.borrow().get(arg.id.get_name())),
                    );
                }
            }

            let mut builder = Builder::new(&mut component, &calyx_ast.lib).validate();
            for (i, arg) in sig.args.iter().enumerate() {
                if let Type::Array {
                    element_type,
                    dims,
                    ports,
                } = &tcx.types[tcx.value_type_map[&arg.id]]
                {
                    let structure =
                        emit_array_decl(element_type, dims, ports, true, &mut builder, tcx)?;

                    if let Structure::Cell(cell) = &structure {
                        arg_ids[i] = (cell.borrow().name(), true);
                    } else {
                        unreachable!("emit_array_decl should return a cell");
                    }

                    env.value_map.insert(arg.id, structure);
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

            env.func_arg_map.insert(sig.name, arg_ids);
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

    let mut env = Env::default();
    let mut main_component = Component::new("main", vec![], true, false, None);
    let mut builder = Builder::new(&mut main_component, &calyx_ast.lib).validate();

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
