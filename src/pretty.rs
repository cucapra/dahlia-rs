use std::borrow::Cow;

use pretty::RcDoc;

use crate::ast::{
    AssignOp, Backend, Command, CommandId, Context, Decl, Def, DimSpec, Expr, ExprId, ForRange,
    IdResolve, Include, InfixOp, Program, Suffix, Type, TypeId, View,
};

type Doc<'a> = RcDoc<'a, ()>;

fn text<'a>(s: impl Into<Cow<'a, str>>) -> Doc<'a> {
    RcDoc::text(s)
}

fn parens<'a>(d: Doc<'a>) -> Doc<'a> {
    text("(").append(d).append(text(")"))
}
fn brackets<'a>(d: Doc<'a>) -> Doc<'a> {
    text("[").append(d).append(text("]"))
}
fn angles<'a>(d: Doc<'a>) -> Doc<'a> {
    text("<").append(d).append(text(">"))
}
fn braces<'a>(d: Doc<'a>) -> Doc<'a> {
    text("{").append(d).append(text("}"))
}
fn quote<'a>(d: Doc<'a>) -> Doc<'a> {
    text("\"").append(d).append(text("\""))
}

fn scope<'a>(doc: Doc<'a>) -> Doc<'a> {
    text("{")
        .append(Doc::hardline().append(doc).nest(2))
        .append(Doc::hardline())
        .append(text("}"))
}

fn emit_id<'a, I: IdResolve>(id: I, context: &'a Context) -> Doc<'a> {
    text(id.resolve_id(&context.ast))
}

pub fn pretty_program<'a>(program: &'a Program, context: &'a Context) -> String {
    let mut sections: Vec<Doc<'a>> = Vec::new();

    if !program.includes.is_empty() {
        sections.push(Doc::intersperse(
            program.includes.iter().map(|i| emit_include(i, context)),
            Doc::hardline(),
        ));
    }
    if !program.defs.is_empty() {
        sections.push(Doc::intersperse(
            program.defs.iter().map(|d| emit_def(d, true, context)),
            Doc::hardline(),
        ));
    }
    let decors = program.decors.as_slice(&context.ast.command_lists);
    if !decors.is_empty() {
        sections.push(Doc::intersperse(
            decors.iter().map(|c| emit_cmd(*c, context)),
            Doc::hardline(),
        ));
    }
    if !program.decls.is_empty() {
        sections.push(Doc::intersperse(
            program.decls.iter().map(|d| {
                text("decl")
                    .append(Doc::space())
                    .append(emit_decl(d, context))
                    .append(text(";"))
            }),
            Doc::hardline(),
        ));
    }
    if !matches!(context.ast.commands[program.cmd], Command::Empty) {
        sections.push(emit_cmd(program.cmd, context));
    }

    Doc::intersperse(sections, Doc::hardline())
        .pretty(usize::MAX)
        .to_string()
}

fn emit_include<'a>(include: &'a Include, context: &'a Context) -> Doc<'a> {
    let backends = Doc::intersperse(
        include
            .backends
            .iter()
            .map(|(b, s)| text(backend_name(b)).append(parens(quote(text(s.as_str()))))),
        Doc::space(),
    );

    text("import")
        .append(Doc::space())
        .append(backends)
        .append(Doc::space())
        .append(scope(Doc::intersperse(
            include.defs.iter().map(|d| emit_def(d, false, context)),
            Doc::hardline(),
        )))
}

fn emit_def<'a>(def: &'a Def, include_body: bool, context: &'a Context) -> Doc<'a> {
    match def {
        Def::Func { sig, body } => {
            let ret_doc = match &context.tcx.types[sig.ret_ty] {
                Type::Void => Doc::nil(),
                _ => text(":")
                    .append(Doc::space())
                    .append(emit_type(sig.ret_ty, context)),
            };

            let args = Doc::intersperse(
                sig.args.iter().map(|d| emit_decl(d, context)),
                text(", "),
            );

            let head = text("def")
                .append(Doc::space())
                .append(emit_id(sig.name, context))
                .append(parens(args))
                .append(Doc::space())
                .append(ret_doc);

            let tail = if include_body {
                Doc::space()
                    .append(text("="))
                    .append(Doc::space())
                    .append(scope(emit_cmd(*body, context)))
            } else {
                text(";")
            };

            head.append(tail)
        }
        Def::Record { name, fields } => {
            let mut entries: Vec<_> = fields.iter().collect();
            entries.sort_by_key(|(f, _)| f.resolve_id(&context.ast));

            let body = Doc::intersperse(
                entries.into_iter().map(|(f, t)| {
                    emit_id(*f, context)
                        .append(text(":"))
                        .append(Doc::space())
                        .append(emit_type(*t, context))
                        .append(text(";"))
                }),
                Doc::hardline(),
            );

            text("record")
                .append(Doc::space())
                .append(emit_id(*name, context))
                .append(Doc::space())
                .append(scope(body))
        }
    }
}

fn emit_decl<'a>(decl: &'a Decl, context: &'a Context) -> Doc<'a> {
    emit_id(decl.id, context)
        .append(text(":"))
        .append(Doc::space())
        .append(emit_type(decl.ty, context))
}

fn emit_type<'a>(ty: TypeId, context: &'a Context) -> Doc<'a> {
    match &context.tcx.types[ty] {
        Type::Float => text("float"),
        Type::Double => text("double"),
        Type::Bool => text("bool"),
        Type::Void => text("void"),
        Type::Rational(_) => text("rational"),
        Type::Bit { length, unsigned } => {
            let prefix = if *unsigned { text("u") } else { Doc::nil() };
            prefix
                .append(text("bit"))
                .append(angles(text(length.to_string())))
        }
        Type::Fixed {
            length_total,
            length_int,
            unsigned,
        } => {
            let prefix = if *unsigned { text("u") } else { Doc::nil() };
            prefix.append(text("fix")).append(angles(
                text(length_total.to_string())
                    .append(text(","))
                    .append(text(length_int.to_string())),
            ))
        }
        Type::StaticInt(v) => text("static").append(parens(text(v.to_string()))),
        Type::Index { static_, dynamic } => {
            let (s0, s1) = static_;
            let (d0, d1) = dynamic;
            let static_range = parens(
                text(s0.to_string())
                    .append(text(", "))
                    .append(text(s1.to_string())),
            );
            let dynamic_range = parens(
                text(d0.to_string())
                    .append(text(", "))
                    .append(text(d1.to_string())),
            );
            text("idx").append(parens(
                static_range.append(text(", ")).append(dynamic_range),
            ))
        }
        Type::Alias(name) => emit_id(*name, context),
        Type::RecType { name, .. } => emit_id(*name, context),
        Type::Array {
            element_type,
            dims,
            ports,
        } => {
            let port_doc = if *ports > 1 {
                braces(text(ports.to_string()))
            } else {
                Doc::nil()
            };

            let dim_docs = Doc::concat(dims.iter().map(|DimSpec { length, bank }| {
                if *bank > 1 {
                    brackets(
                        text(length.to_string())
                            .append(Doc::space())
                            .append(text("bank"))
                            .append(Doc::space())
                            .append(text(bank.to_string())),
                    )
                } else {
                    brackets(text(length.to_string()))
                }
            }));

            emit_type(*element_type, context)
                .append(port_doc)
                .append(dim_docs)
        }
        Type::Func { args, ret } => {
            let args = args
                .as_slice(&context.tcx.type_lists)
                .iter()
                .map(|a| emit_type(*a, context));

            Doc::intersperse(args, text("->"))
                .append(text(" -> "))
                .append(emit_type(*ret, context))
        }
    }
}

fn emit_base_int<'a>(value: i64, base: u8) -> Doc<'a> {
    match base {
        8 => text(format!("0{value:o}")),
        16 => text(format!("0x{value:x}")),
        _ => text(value.to_string()),
    }
}

fn emit_expr<'a>(expr: ExprId, context: &'a Context) -> Doc<'a> {
    match &context.ast.exprs[expr] {
        Expr::Placeholder => {
            panic!("Placeholder expressions should not be present in the final AST")
        }
        Expr::Cast { expr, ty } => parens(
            emit_expr(*expr, context)
                .append(Doc::space())
                .append(text("as"))
                .append(Doc::space())
                .append(emit_type(*ty, context)),
        ),
        Expr::Application { func, args } => {
            let arg_docs = args
                .as_slice(&context.ast.expr_lists)
                .iter()
                .map(|e| emit_expr(*e, context));
            emit_id(*func, context).append(parens(Doc::intersperse(arg_docs, text(", "))))
        }
        Expr::IntLiteral { value, base } => emit_base_int(*value, *base),
        Expr::RationalLiteral(d) => text(d.as_str()),
        Expr::BoolLiteral(b) => text(if *b { "true" } else { "false" }),
        Expr::Id(v) => emit_id(*v, context),
        Expr::BinOp { left, op, right } => parens(
            emit_expr(*left, context)
                .append(Doc::space())
                .append(text(infix_op_str(*op)))
                .append(Doc::space())
                .append(emit_expr(*right, context)),
        ),
        Expr::ArrayAccess { array, indices } => emit_id(*array, context).append(Doc::concat(
            indices
                .as_slice(&context.ast.expr_lists)
                .iter()
                .map(|i| brackets(emit_expr(*i, context))),
        )),
        Expr::ArrayLiteral(elements) => braces(Doc::intersperse(
            elements
                .as_slice(&context.ast.expr_lists)
                .iter()
                .map(|e| emit_expr(*e, context)),
            text(", "),
        )),
        Expr::RecordAccess { record, field } => emit_expr(*record, context)
            .append(text("."))
            .append(emit_id(*field, context)),
        Expr::RecordLiteral(fields) => scope(Doc::intersperse(
            fields.iter().map(|(f, e)| {
                emit_id(*f, context)
                    .append(Doc::space())
                    .append(text("="))
                    .append(Doc::space())
                    .append(emit_expr(*e, context))
                    .append(text(";"))
            }),
            Doc::space(),
        )),
    }
}

fn emit_range<'a>(range: &'a ForRange, context: &'a Context) -> Doc<'a> {
    let typ_annot = range.ty.map_or_else(Doc::nil, |t| {
        text(":")
            .append(Doc::space())
            .append(emit_type(t, context))
    });

    let rev = if range.rev {
        text("rev").append(Doc::space())
    } else {
        Doc::nil()
    };

    let inner = text("let")
        .append(Doc::space())
        .append(emit_id(range.iter, context))
        .append(typ_annot)
        .append(Doc::space())
        .append(text("="))
        .append(Doc::space())
        .append(rev)
        .append(text(range.start.to_string()))
        .append(Doc::space())
        .append(text(".."))
        .append(Doc::space())
        .append(text(range.end.to_string()));

    let head = parens(inner);

    if range.unroll > 1 {
        head.append(Doc::space())
            .append(text("unroll"))
            .append(Doc::space())
            .append(text(range.unroll.to_string()))
    } else {
        head
    }
}

fn emit_view<'a>(view: &'a View, context: &'a Context) -> Doc<'a> {
    let suf = match &view.suffix {
        Suffix::Aligned { factor, e } => text(factor.to_string())
            .append(Doc::space())
            .append(text("*"))
            .append(Doc::space())
            .append(emit_expr(*e, context)),
        Suffix::Rotation(e) => emit_expr(*e, context).append(text("!")),
    };

    let prefix = view.prefix.map_or_else(Doc::nil, |p| {
        Doc::space()
            .append(text("+"))
            .append(Doc::space())
            .append(text(p.to_string()))
    });

    let shrink = view.shrink.map_or_else(Doc::nil, |s| {
        Doc::space()
            .append(text("bank"))
            .append(Doc::space())
            .append(text(s.to_string()))
    });

    suf.append(Doc::space())
        .append(text(":"))
        .append(prefix)
        .append(shrink)
}

fn emit_cmd<'a>(cmd: CommandId, context: &'a Context) -> Doc<'a> {
    match &context.ast.commands[cmd] {
        Command::Empty => Doc::nil(),
        Command::Block(c) => scope(emit_cmd(*c, context)),
        Command::Par(cmds) => Doc::intersperse(
            cmds.iter().map(|c| emit_cmd(*c, context)),
            Doc::hardline(),
        ),
        Command::Seq(cmds) => Doc::intersperse(
            cmds.iter().map(|c| emit_cmd(*c, context)),
            Doc::hardline().append(text("---")).append(Doc::hardline()),
        ),
        Command::Let { id, ty, value } => {
            let typ_annot = ty.map_or_else(Doc::nil, |t| {
                text(":")
                    .append(Doc::space())
                    .append(emit_type(t, context))
            });
            let val = value.map_or_else(Doc::nil, |e| {
                Doc::space()
                    .append(text("="))
                    .append(Doc::space())
                    .append(emit_expr(e, context))
            });
            text("let")
                .append(Doc::space())
                .append(emit_id(*id, context))
                .append(typ_annot)
                .append(val)
                .append(text(";"))
        }
        Command::Update { lhs, op, rhs } => emit_expr(*lhs, context)
            .append(Doc::space())
            .append(text(assign_op_str(op)))
            .append(Doc::space())
            .append(emit_expr(*rhs, context))
            .append(text(";")),
        Command::View { id, arr_id, dims } => {
            let dim_docs = Doc::concat(dims.iter().map(|v| brackets(emit_view(v, context))));
            text("view")
                .append(Doc::space())
                .append(emit_id(*id, context))
                .append(Doc::space())
                .append(text("="))
                .append(Doc::space())
                .append(emit_id(*arr_id, context))
                .append(Doc::space())
                .append(dim_docs)
                .append(text(";"))
        }
        Command::Split { id, arr_id, dims } => {
            let dim_docs = Doc::concat(dims.iter().map(|f| {
                brackets(text("by").append(Doc::space()).append(text(f.to_string())))
            }));
            text("split")
                .append(Doc::space())
                .append(emit_id(*id, context))
                .append(Doc::space())
                .append(text("="))
                .append(Doc::space())
                .append(emit_id(*arr_id, context))
                .append(dim_docs)
                .append(text(";"))
        }
        Command::Return(e) => text("return")
            .append(Doc::space())
            .append(emit_expr(*e, context))
            .append(text(";")),
        Command::IfElse { cond, then, else_ } => {
            let head = text("if")
                .append(Doc::space())
                .append(parens(emit_expr(*cond, context)))
                .append(Doc::space())
                .append(scope(emit_cmd(*then, context)));

            match &context.ast.commands[*else_] {
                Command::Empty => head,
                Command::IfElse { .. } => head
                    .append(Doc::space())
                    .append(text("else"))
                    .append(Doc::space())
                    .append(emit_cmd(*else_, context)),
                _ => head
                    .append(Doc::space())
                    .append(text("else"))
                    .append(Doc::space())
                    .append(scope(emit_cmd(*else_, context))),
            }
        }
        Command::While {
            cond,
            pipeline,
            body,
        } => {
            let pipe = if *pipeline {
                Doc::space().append(text("pipeline"))
            } else {
                Doc::nil()
            };
            text("while")
                .append(Doc::space())
                .append(parens(emit_expr(*cond, context)))
                .append(pipe)
                .append(Doc::space())
                .append(scope(emit_cmd(*body, context)))
        }
        Command::For {
            range,
            pipeline,
            body,
            combine,
        } => {
            let pipe = if *pipeline {
                Doc::space().append(text("pipeline"))
            } else {
                Doc::nil()
            };
            let head = text("for")
                .append(Doc::space())
                .append(emit_range(range, context))
                .append(pipe)
                .append(Doc::space())
                .append(scope(emit_cmd(*body, context)));

            match &context.ast.commands[*combine] {
                Command::Empty => head,
                _ => head
                    .append(Doc::space())
                    .append(text("combine"))
                    .append(Doc::space())
                    .append(scope(emit_cmd(*combine, context))),
            }
        }
        Command::Decorate(s) => text("decor")
            .append(Doc::space())
            .append(quote(text(s.as_str()))),
        Command::Expr(e) => emit_expr(*e, context).append(text(";")),
    }
}

fn infix_op_str(op: InfixOp) -> &'static str {
    match op {
        InfixOp::Mul => "*",
        InfixOp::Div => "/",
        InfixOp::Mod => "%",
        InfixOp::Add => "+",
        InfixOp::Sub => "-",
        InfixOp::Shl => "<<",
        InfixOp::Shr => ">>",
        InfixOp::Eq => "==",
        InfixOp::Neq => "!=",
        InfixOp::Le => "<=",
        InfixOp::Ge => ">=",
        InfixOp::Lt => "<",
        InfixOp::Gt => ">",
        InfixOp::And => "&&",
        InfixOp::Or => "||",
        InfixOp::Band => "&",
        InfixOp::Bor => "|",
        InfixOp::Bxor => "^",
    }
}

fn assign_op_str(op: &AssignOp) -> &'static str {
    match op {
        AssignOp::Assign => ":=",
        AssignOp::AddAssign => "+=",
        AssignOp::SubAssign => "-=",
        AssignOp::MulAssign => "*=",
        AssignOp::DivAssign => "/=",
    }
}

fn backend_name(backend: &Backend) -> &'static str {
    match backend {
        Backend::Cpp => "c++",
        Backend::Vivado => "vivado",
        Backend::Futil | Backend::Calyx => "calyx",
    }
}
