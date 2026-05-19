use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use bigdecimal::BigDecimal;
use bigdecimal::Num;
use bigdecimal::ToPrimitive;
use calyx_ir::Attributes;
use calyx_ir::BoolAttr;
use calyx_ir::Component;
use calyx_ir::Id;
use calyx_ir::Invoke;
use calyx_ir::PortDef;
use calyx_ir::{Builder, Cell, Control, Port, RRC};
use cranelift_entity::EntityList;
use indexmap::IndexMap;

use crate::ast::AssignOp;
use crate::ast::Command;
use crate::ast::DimSpec;
use crate::ast::Expr;
use crate::ast::FuncId;
use crate::ast::IdResolve;
use crate::ast::InfixOp;
use crate::ast::Type;
use crate::ast::TypeId;
use crate::ast::{Ast, CommandId, Decl, Def, ExprId, Program, TypeContext, ValueId};
use crate::utils::bits_needed;

type Guard = calyx_ir::Guard<calyx_ir::Nothing>;
type Assignment = calyx_ir::Assignment<calyx_ir::Nothing>;

struct ExprEmitOutput {
    port: RRC<Port>,
    done: Option<RRC<Port>>,
    assignments: Vec<Assignment>,
}

impl ExprEmitOutput {
    fn new(port: RRC<Port>, done: Option<RRC<Port>>, assignments: Vec<Assignment>) -> Self {
        Self {
            port,
            done,
            assignments,
        }
    }
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
            ref_cells.push((id.clone(), emit_output.port.borrow_mut().cell_parent()));
        } else {
            inputs.push((id.clone(), emit_output.port));
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
    rhs_done: RRC<Port>,
    env: &mut Env,
    builder: &mut Builder,
    ast: &Ast,
    tcx: &TypeContext,
) -> Result<ExprEmitOutput> {
    match &ast.exprs[expr] {
        Expr::Id(id) => {
            if let Structure::Cell(cell) =
                env.value_map.get(id).expect("def should be in value_map")
            {
                let cell_in = cell.borrow().get("in");

                let cell_write_en = cell.borrow().get("write_en");
                let assgn = builder.build_assignment(cell_write_en, rhs_done, Guard::True);

                let cell_done = cell.borrow().get("done");

                Ok(ExprEmitOutput::new(cell_in, Some(cell_done), vec![assgn]))
            } else {
                bail!("Cannot assign to non-cell");
            }
        }
        Expr::ArrayAccess { array, indices } => {
            let cell = if let Structure::Cell(cell) = env
                .value_map
                .get(array)
                .expect("def should be in value_map")
            {
                cell.clone()
            } else {
                unreachable!("Arrays should be emitted as cells")
            };

            let mem_write_en = cell.borrow().get("write_en");
            let mem_done = cell.borrow().get("done");
            let mem_write_data = cell.borrow().get("write_data");
            let mem_content_en = cell.borrow().get("content_en");

            let mut assignments = Vec::new();

            for (i, idx) in indices.as_slice(&ast.expr_lists).iter().enumerate() {
                let idx_out = emit_expr(*idx, env, builder, ast, tcx).with_context(|| {
                    format!(
                        "failed to emit index for array {} access",
                        array.resolve_id(ast)
                    )
                })?;
                assignments.extend(idx_out.assignments.into_iter());
                assignments.push(builder.build_assignment(
                    cell.borrow().get(format!("addr{}", i)),
                    idx_out.port,
                    Guard::True,
                ));
            }

            let const1 = builder.add_constant(1, 1).borrow().get("out");

            assignments.push(builder.build_assignment(mem_content_en, const1, Guard::True));
            assignments.push(builder.build_assignment(mem_write_en, rhs_done, Guard::True));

            Ok(ExprEmitOutput::new(
                mem_write_data,
                Some(mem_done),
                assignments,
            ))
        }
        _ => unreachable!("lvalue should be either an Id or an ArrayAccess"),
    }
}

fn infix_op_to_primitive(op: InfixOp) -> &'static str {
    match op {
        InfixOp::Add => "add",
        InfixOp::Sub => "sub",
        InfixOp::Mul => "mult_pipe",
        InfixOp::Div => "div_pipe",
        InfixOp::Mod => "div_pipe",
        InfixOp::Lt => "lt",
        InfixOp::Gt => "gt",
        InfixOp::Le => "le",
        InfixOp::Ge => "ge",
        InfixOp::Neq => "neq",
        InfixOp::Eq => "eq",
        InfixOp::And => "and",
        InfixOp::Or => "or",
        InfixOp::Band => "and",
        InfixOp::Bor => "or",
        InfixOp::Shl => "lsh",
        InfixOp::Shr => "rsh",
        InfixOp::Bxor => "xor",
    }
}

fn signed(ty: TypeId, op: InfixOp, tcx: &TypeContext) -> bool {
    match op {
        InfixOp::Add | InfixOp::Sub | InfixOp::Mul | InfixOp::Div | InfixOp::Mod => {
            match &tcx.types[ty] {
                Type::Bit { unsigned, .. } | Type::Fixed { unsigned, .. } => !unsigned,
                _ => false,
            }
        }
        _ => false,
    }
}

fn emit_expr(
    expr: ExprId,
    env: &mut Env,
    builder: &mut Builder,
    ast: &Ast,
    tcx: &TypeContext,
) -> Result<ExprEmitOutput> {
    match &ast.exprs[expr] {
        Expr::IntLiteral { .. } => {
            bail!("Cannot emit unannotated constants; wrap in a cast expression")
        }
        Expr::Application { .. } => {
            bail!("Function application should be assigned to a let binding")
        }
        Expr::RationalLiteral(..) => {
            bail!("Cannot emit unannotated constants; cast to fixed-point first")
        }
        Expr::Id(id) => {
            match env
                .value_map
                .get(id)
                .expect("variable should be in value_map")
            {
                Structure::Cell(cell) => {
                    Ok(ExprEmitOutput::new(cell.borrow().get("out"), None, vec![]))
                }
                Structure::Port(port) => Ok(ExprEmitOutput::new(port.clone(), None, vec![])),
            }
        }
        Expr::BoolLiteral(value) => {
            let const_cell = builder.add_primitive("bool_const", "std_const", &[1, *value as u64]);

            Ok(ExprEmitOutput::new(
                const_cell.borrow().get("out"),
                None,
                vec![],
            ))
        }
        Expr::Cast { expr, ty } => {
            match &ast.exprs[*expr] {
                Expr::IntLiteral { value, .. } => {
                    let (width, _) = bits_for_type(*ty, tcx);

                    // how does this handle negative int literals?
                    let const_cell =
                        builder.add_primitive("const", "std_const", &[width, *value as u64]);
                    Ok(ExprEmitOutput::new(
                        const_cell.borrow().get("out"),
                        None,
                        vec![],
                    ))
                }
                Expr::RationalLiteral(value) => {
                    let (width, Some(int_width)) = bits_for_type(*ty, tcx) else {
                        unreachable!(
                            "rational expression should only be casted to a fixed-point type"
                        );
                    };

                    let frac_width = width - int_width;
                    let scaled_value = value
                        .parse::<BigDecimal>()
                        .expect("invalid rational literal")
                        * BigDecimal::from(1u64 << frac_width);

                    // should we do rounding at all, or simply bail if the value cannot be represented as in the Scala version?
                    let rounded_value =
                        scaled_value.with_scale_round(0, bigdecimal::RoundingMode::HalfEven);

                    let twos_complement = rounded_value
                        .to_i64()
                        .ok_or_else(|| anyhow!("rational literal cannot be represented in i64"))?
                        as u64;

                    let cell = builder.add_primitive(
                        "float_const",
                        "std_const",
                        &[width, twos_complement],
                    );

                    Ok(ExprEmitOutput::new(cell.borrow().get("out"), None, vec![]))
                }
                _ => {
                    let (value_width, _) = bits_for_type(tcx.expr_type_map[expr], tcx);
                    let (type_width, _) = bits_for_type(*ty, tcx);

                    let value_out = emit_expr(*expr, env, builder, ast, tcx)
                        .context("failed to emit cast expression value")?;

                    if value_width == type_width {
                        Ok(value_out)
                    } else {
                        let cell = if type_width > value_width {
                            builder.add_primitive("pad", "std_pad", &[value_width, type_width])
                        } else {
                            builder.add_primitive("slice", "std_slice", &[value_width, type_width])
                        };

                        let mut assignments = value_out.assignments;
                        assignments.push(builder.build_assignment(
                            cell.borrow().get("in"),
                            value_out.port,
                            Guard::True,
                        ));

                        Ok(ExprEmitOutput::new(
                            cell.borrow().get("out"),
                            value_out.done,
                            assignments,
                        ))
                    }
                }
            }
        }
        Expr::BinOp { left, op, right } => {
            let op_string = infix_op_to_primitive(*op);
            let slow_op = matches!(op, InfixOp::Mul | InfixOp::Div | InfixOp::Mod);

            let out_port = match op {
                InfixOp::Div => "out_quotient",
                InfixOp::Mod => "out_remainder",
                _ => "out",
            };

            let lhs_out =
                emit_expr(*left, env, builder, ast, tcx).context("failed to emit LHS of binop")?;
            let rhs_out =
                emit_expr(*right, env, builder, ast, tcx).context("failed to emit RHS of binop")?;

            let (lhs_width, lhs_int_width) = bits_for_type(tcx.expr_type_map[left], tcx);
            let (rhs_width, rhs_int_width) = bits_for_type(tcx.expr_type_map[right], tcx);

            match (lhs_int_width, rhs_int_width) {
                (Some(lhs_int_width), Some(rhs_int_width)) => {
                    if slow_op {
                        if lhs_int_width != rhs_int_width {
                            bail!("Mismatched operand widths for binop");
                        }
                    } else {
                        if !(lhs_width == rhs_width && lhs_int_width == rhs_int_width) {
                            bail!("Mismatched operand widths for binop");
                        }
                    }
                }
                (None, None) => {
                    if lhs_width != rhs_width {
                        bail!("Mismatched operand widths for binop");
                    }
                }
                _ => {
                    bail!("Cannot mix fixed-point and non-fixed-point types in binop");
                }
            };

            let signed_str = if signed(tcx.expr_type_map[left], *op, tcx) {
                "s"
            } else {
                ""
            };

            let cell = if let Some(lhs_int_width) = lhs_int_width {
                let lhs_frac_width = lhs_width - lhs_int_width;
                builder.add_primitive(
                    op_string,
                    format!("std_fp_{}{}", signed_str, op_string),
                    &[lhs_width, lhs_int_width, lhs_frac_width],
                )
            } else {
                builder.add_primitive(
                    op_string,
                    format!("std_{}{}", signed_str, op_string),
                    &[lhs_width],
                )
            };

            let mut assignments = Vec::new();

            assignments.extend(lhs_out.assignments.into_iter());
            assignments.extend(rhs_out.assignments.into_iter());

            let cell_done = cell.borrow().get("done");
            if slow_op {
                let cell_go = cell.borrow().get("go");
                let const1 = builder.add_constant(1, 1).borrow().get("out");
                assignments.push(builder.build_assignment(
                    cell_go,
                    const1,
                    Guard::Not(Guard::Port(cell_done.clone()).into()),
                ));
            }

            assignments.push(builder.build_assignment(
                cell.borrow().get("left"),
                lhs_out.port,
                Guard::True,
            ));
            assignments.push(builder.build_assignment(
                cell.borrow().get("right"),
                rhs_out.port,
                Guard::True,
            ));

            Ok(ExprEmitOutput::new(
                cell.borrow().get(out_port),
                if slow_op { Some(cell_done) } else { None },
                assignments,
            ))
        }
        Expr::ArrayAccess { array, indices } => {
            let cell = if let Structure::Cell(cell) = env
                .value_map
                .get(array)
                .expect("def should be in value_map")
            {
                cell.clone()
            } else {
                unreachable!("Arrays should be emitted as cells")
            };

            let mem_write_en = cell.borrow().get("write_en");
            let mem_read_data = cell.borrow().get("read_data");
            let mem_done = cell.borrow().get("done");
            let mem_content_en = cell.borrow().get("content_en");

            let mut assignments = Vec::new();

            for (i, idx) in indices.as_slice(&ast.expr_lists).iter().enumerate() {
                let idx_out = emit_expr(*idx, env, builder, ast, tcx).with_context(|| {
                    format!(
                        "failed to emit index for array {} access",
                        array.resolve_id(ast)
                    )
                })?;
                assignments.extend(idx_out.assignments.into_iter());
                assignments.push(builder.build_assignment(
                    cell.borrow().get(format!("addr{}", i)),
                    idx_out.port,
                    Guard::True,
                ));
            }

            let const0 = builder.add_constant(0, 1).borrow().get("out");
            let const1 = builder.add_constant(1, 1).borrow().get("out");

            assignments.push(builder.build_assignment(mem_content_en, const1, Guard::True));
            assignments.push(builder.build_assignment(mem_write_en, const0, Guard::True));

            Ok(ExprEmitOutput::new(
                mem_read_data,
                Some(mem_done),
                assignments,
            ))
        }
        _ => unimplemented!("expression type not supported yet: {:?}", ast.exprs[expr]),
    }
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
                    let structure =
                        emit_array_decl(element_type, dims, ports, false, false, builder, tcx)
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
                    let out_out = out.port;

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
                    let out_out = out.port;

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
        Command::Expr(expr) => {
            if let Expr::Application { func, args } = &ast.exprs[*expr] {
                let (_, assignments, invoke_control) =
                    emit_invoke(*func, *args, env, builder, ast, tcx).with_context(|| {
                        format!(
                            "failed to emit function application for {}",
                            func.resolve_id(ast)
                        )
                    })?;
                builder.add_continuous_assignments(assignments);
                Ok(invoke_control)
            } else {
                bail!("Unsupported expression command")
            }
        }
        Command::Update { lhs, op, rhs } => {
            if *op == AssignOp::Assign {
                let rhs_out =
                    emit_expr(*rhs, env, builder, ast, tcx).context("failed to emit update RHS")?;

                let rhs_done = rhs_out
                    .done
                    .unwrap_or_else(|| builder.add_constant(1, 1).borrow().get("out"));

                let lhs_out = emit_lvalue(*lhs, rhs_done, env, builder, ast, tcx)
                    .context("failed to emit update LHS")?;

                assert!(lhs_out.done.is_some(), "Lvalue should have a done signal");

                let group = builder.add_group("update");
                group
                    .borrow_mut()
                    .assignments
                    .extend(lhs_out.assignments.into_iter());
                group
                    .borrow_mut()
                    .assignments
                    .extend(rhs_out.assignments.into_iter());

                group
                    .borrow_mut()
                    .assignments
                    .push(builder.build_assignment(lhs_out.port, rhs_out.port, Guard::True));

                let group_done = group.borrow().get("done");
                let lhs_done = lhs_out.done.unwrap();
                group
                    .borrow_mut()
                    .assignments
                    .push(builder.build_assignment(group_done, lhs_done, Guard::True));

                Ok(Control::enable(group))
            } else {
                bail!("Only support simple assignment for now")
            }
        }
        Command::IfElse { cond, then, else_ } => {
            let cond_out = emit_expr(*cond, env, builder, ast, tcx)
                .context("failed to emit condition for if command")?;

            let then_control = emit_command(*then, env, builder, ast, tcx)
                .context("failed to emit then branch")?;
            let else_control = emit_command(*else_, env, builder, ast, tcx)
                .context("failed to emit else branch")?;

            if let Some(done) = cond_out.done {
                let group = builder.add_group("cond");

                let cond_done = group.borrow().get("done");
                group.borrow_mut().assignments = cond_out.assignments;
                group
                    .borrow_mut()
                    .assignments
                    .push(builder.build_assignment(cond_done, done, Guard::True));

                Ok(Control::seq(vec![
                    Control::enable(group),
                    Control::if_(
                        cond_out.port,
                        None,
                        then_control.into(),
                        else_control.into(),
                    ),
                ]))
            } else {
                let group = builder.add_comb_group("cond");

                group.borrow_mut().assignments = cond_out.assignments;

                Ok(Control::if_(
                    cond_out.port,
                    Some(group),
                    then_control.into(),
                    else_control.into(),
                ))
            }
        }
        Command::Decorate(..) | Command::Empty => Ok(Control::empty()),
        Command::For { .. } => {
            bail!("For loops should have been lowered")
        }
        Command::While { cond, body, .. } => {
            let body_control = emit_command(*body, env, builder, ast, tcx)
                .context("failed to emit while loop body")?;

            let cond_out = emit_expr(*cond, env, builder, ast, tcx)
                .context("failed to emit condition for while loop")?;

            if cond_out.done.is_some() {
                bail!("While loop conditions should be combinatorial");
            }

            let group = builder.add_comb_group("cond");
            group.borrow_mut().assignments = cond_out.assignments;

            Ok(Control::while_(
                cond_out.port,
                Some(group),
                body_control.into(),
            ))
        }
        Command::Return(expr) => {
            if let Expr::Id(_) = &ast.exprs[*expr] {
                let mut out = emit_expr(*expr, env, builder, ast, tcx)
                    .context("failed to emit return expression")?;

                let this_out = builder.component.signature.borrow().get("out");
                let out_out = out.port;

                let assign = builder.build_assignment(this_out, out_out, Guard::True);
                out.assignments.push(assign);

                builder.add_continuous_assignments(out.assignments);

                Ok(Control::empty())
            } else {
                bail!("Can only return a variable")
            }
        }
        Command::View { .. } | Command::Split { .. } => {
            bail!("View and Split should have been lowered")
        }
    }
}

fn emit_array_decl(
    element_type: &TypeId,
    dims: &Vec<DimSpec>,
    ports: &usize,
    external: bool,
    reference: bool,
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

    if external {
        cell.borrow_mut().add_attribute(BoolAttr::External, 1);
    }
    cell.borrow_mut().set_reference(reference);

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
        } => emit_array_decl(element_type, dims, ports, true, false, builder, tcx)?,
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
                        emit_array_decl(element_type, dims, ports, false, true, &mut builder, tcx)?;

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
