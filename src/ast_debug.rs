#![allow(dead_code)]

use crate::ast::{
    AssignOp, Backend, Command, CommandId, Context, Decl, Def, DimSpec, Expr, ExprId, ForRange,
    FuncSig, Id, Include, InfixOp, Program, Suffix, Type, TypeId, View,
};

#[derive(Debug)]
enum DebugType<'a> {
    Float,
    Double,
    Bool,
    Bit {
        length: usize,
        unsigned: bool,
    },
    Fixed {
        length_total: usize,
        length_int: usize,
        unsigned: bool,
    },
    Alias(&'a Id),
    Array {
        element_type: Box<DebugType<'a>>,
        dims: &'a [DimSpec],
        ports: usize,
    },
    StaticInt(u64),
    Index {
        static_: (i64, i64),
        dynamic: (i64, i64),
    },
    Void,
    Rational(&'a str),
    Func {
        args: Vec<DebugType<'a>>,
        ret: Box<DebugType<'a>>,
    },
    RecType {
        name: &'a Id,
        fields: Vec<(&'a Id, DebugType<'a>)>,
    },
}

#[derive(Debug)]
enum DebugExpr<'a> {
    Cast {
        expr: Box<DebugExpr<'a>>,
        ty: DebugType<'a>,
    },

    ArrayLiteral(Vec<DebugExpr<'a>>),
    RecordLiteral(Vec<(&'a Id, DebugExpr<'a>)>),

    RationalLiteral(&'a str),
    IntLiteral {
        value: i64,
        base: u8,
    },
    BoolLiteral(bool),

    ArrayAccess {
        array: &'a Id,
        indices: Vec<DebugExpr<'a>>,
    },
    RecordAccess {
        record: Box<DebugExpr<'a>>,
        field: &'a Id,
    },

    Application {
        func: &'a Id,
        args: Vec<DebugExpr<'a>>,
    },

    Id(&'a Id),

    BinOp {
        left: Box<DebugExpr<'a>>,
        op: &'a InfixOp,
        right: Box<DebugExpr<'a>>,
    },
}

#[derive(Debug)]
struct DebugForRange<'a> {
    id: &'a Id,
    ty: Option<DebugType<'a>>,
    rev: bool,
    start: usize,
    end: usize,
    unroll: usize,
}

#[derive(Debug)]
enum DebugCommand<'a> {
    Empty,
    Par(Vec<DebugCommand<'a>>),
    Seq(Vec<DebugCommand<'a>>),
    Let {
        id: &'a Id,
        ty: Option<DebugType<'a>>,
        value: Option<DebugExpr<'a>>,
    },
    Update {
        lhs: DebugExpr<'a>,
        op: &'a AssignOp,
        rhs: DebugExpr<'a>,
    },
    View {
        id: &'a Id,
        arr_id: &'a Id,
        dims: Vec<DebugView<'a>>,
    },
    Split {
        id: &'a Id,
        arr_id: &'a Id,
        dims: &'a [usize],
    },
    Return(DebugExpr<'a>),
    IfElse {
        cond: DebugExpr<'a>,
        then: Box<DebugCommand<'a>>,
        else_: Box<DebugCommand<'a>>,
    },
    While {
        cond: DebugExpr<'a>,
        pipeline: bool,
        body: Box<DebugCommand<'a>>,
    },
    For {
        range: DebugForRange<'a>,
        pipeline: bool,
        body: Box<DebugCommand<'a>>,
        combine: Box<DebugCommand<'a>>,
    },
    Decorate(&'a str),
    Expr(DebugExpr<'a>),
}

#[derive(Debug)]
struct DebugDecl<'a> {
    id: &'a Id,
    ty: DebugType<'a>,
}

#[derive(Debug)]
struct DebugFuncSig<'a> {
    name: &'a Id,
    args: Vec<DebugDecl<'a>>,
    ret_ty: Option<DebugType<'a>>,
}

#[derive(Debug)]
enum DebugDef<'a> {
    Func {
        sig: DebugFuncSig<'a>,
        body: DebugCommand<'a>,
    },
    Record {
        name: &'a Id,
        fields: Vec<DebugDecl<'a>>,
    },
}

#[derive(Debug)]
enum DebugSuffix<'a> {
    Rotation(DebugExpr<'a>),
    Aligned { factor: usize, e: DebugExpr<'a> },
}

#[derive(Debug)]
struct DebugView<'a> {
    suffix: DebugSuffix<'a>,
    prefix: Option<usize>,
    shrink: Option<usize>,
}

#[derive(Debug)]
struct DebugInclude<'a> {
    backends: Vec<(&'a Backend, &'a str)>,
    defs: Vec<DebugFuncSig<'a>>,
}

#[derive(Debug)]
struct DebugProgram<'a> {
    includes: Vec<DebugInclude<'a>>,
    defs: Vec<DebugDef<'a>>,
    decors: Vec<DebugCommand<'a>>,
    decls: Vec<DebugDecl<'a>>,
    cmd: DebugCommand<'a>,
}

pub fn ast_debug(context: &Context, program: &Program) {
    println!("{:#?}", debug_program(context, program));
}

fn debug_program<'a>(context: &'a Context, program: &'a Program) -> DebugProgram<'a> {
    DebugProgram {
        includes: program
            .includes
            .iter()
            .map(|include| debug_include(context, include))
            .collect(),
        defs: program
            .defs
            .iter()
            .map(|def| debug_def(context, def))
            .collect(),
        decors: debug_command_list(context, program.decors.as_slice(&context.command_lists)),
        decls: program
            .decls
            .iter()
            .map(|decl| debug_decl(context, decl))
            .collect(),
        cmd: debug_command_id(context, program.cmd),
    }
}

fn debug_include<'a>(context: &'a Context, include: &'a Include) -> DebugInclude<'a> {
    DebugInclude {
        backends: include
            .backends
            .iter()
            .map(|(backend, value)| (backend, value.as_str()))
            .collect(),
        defs: include
            .defs
            .iter()
            .map(|sig| debug_func_sig(context, sig))
            .collect(),
    }
}

fn debug_def<'a>(context: &'a Context, def: &'a Def) -> DebugDef<'a> {
    match def {
        Def::Func { sig, body } => DebugDef::Func {
            sig: debug_func_sig(context, sig),
            body: debug_command_id(context, *body),
        },
        Def::Record { name, fields } => DebugDef::Record {
            name,
            fields: debug_decl_list(context, fields),
        },
    }
}

fn debug_func_sig<'a>(context: &'a Context, sig: &'a FuncSig) -> DebugFuncSig<'a> {
    DebugFuncSig {
        name: &sig.name,
        args: debug_decl_list(context, &sig.args),
        ret_ty: sig.ret_ty.map(|ty| debug_type_id(context, ty)),
    }
}

fn debug_decl_list<'a>(context: &'a Context, decls: &'a [Decl]) -> Vec<DebugDecl<'a>> {
    decls.iter().map(|decl| debug_decl(context, decl)).collect()
}

fn debug_decl<'a>(context: &'a Context, decl: &'a Decl) -> DebugDecl<'a> {
    DebugDecl {
        id: &decl.id,
        ty: debug_type_id(context, decl.ty),
    }
}

fn debug_view<'a>(context: &'a Context, view: &'a View) -> DebugView<'a> {
    DebugView {
        suffix: debug_suffix(context, &view.suffix),
        prefix: view.prefix,
        shrink: view.shrink,
    }
}

fn debug_suffix<'a>(context: &'a Context, suffix: &'a Suffix) -> DebugSuffix<'a> {
    match suffix {
        Suffix::Rotation(expr) => DebugSuffix::Rotation(debug_expr_id(context, *expr)),
        Suffix::Aligned { factor, e } => DebugSuffix::Aligned {
            factor: *factor,
            e: debug_expr_id(context, *e),
        },
    }
}

fn debug_for_range<'a>(context: &'a Context, range: &'a ForRange) -> DebugForRange<'a> {
    DebugForRange {
        id: &range.id,
        ty: range.ty.map(|ty| debug_type_id(context, ty)),
        rev: range.rev,
        start: range.start,
        end: range.end,
        unroll: range.unroll,
    }
}

fn debug_command_id<'a>(context: &'a Context, command: CommandId) -> DebugCommand<'a> {
    debug_command(context, &context.commands[command])
}

fn debug_command_list<'a>(context: &'a Context, commands: &[CommandId]) -> Vec<DebugCommand<'a>> {
    commands
        .iter()
        .map(|command| debug_command_id(context, *command))
        .collect()
}

fn debug_command<'a>(context: &'a Context, command: &'a Command) -> DebugCommand<'a> {
    match command {
        Command::Empty => DebugCommand::Empty,
        Command::Par(commands) => DebugCommand::Par(debug_command_list(context, commands)),
        Command::Seq(commands) => DebugCommand::Seq(debug_command_list(context, commands)),
        Command::Let { id, ty, value } => DebugCommand::Let {
            id,
            ty: ty.map(|ty| debug_type_id(context, ty)),
            value: value.map(|expr| debug_expr_id(context, expr)),
        },
        Command::Update { lhs, op, rhs } => DebugCommand::Update {
            lhs: debug_expr_id(context, *lhs),
            op,
            rhs: debug_expr_id(context, *rhs),
        },
        Command::View { id, arr_id, dims } => DebugCommand::View {
            id,
            arr_id,
            dims: dims.iter().map(|view| debug_view(context, view)).collect(),
        },
        Command::Split { id, arr_id, dims } => DebugCommand::Split { id, arr_id, dims },
        Command::Return(expr) => DebugCommand::Return(debug_expr_id(context, *expr)),
        Command::IfElse { cond, then, else_ } => DebugCommand::IfElse {
            cond: debug_expr_id(context, *cond),
            then: Box::new(debug_command_id(context, *then)),
            else_: Box::new(debug_command_id(context, *else_)),
        },
        Command::While {
            cond,
            pipeline,
            body,
        } => DebugCommand::While {
            cond: debug_expr_id(context, *cond),
            pipeline: *pipeline,
            body: Box::new(debug_command_id(context, *body)),
        },
        Command::For {
            range,
            pipeline,
            body,
            combine,
        } => DebugCommand::For {
            range: debug_for_range(context, range),
            pipeline: *pipeline,
            body: Box::new(debug_command_id(context, *body)),
            combine: Box::new(debug_command_id(context, *combine)),
        },
        Command::Decorate(value) => DebugCommand::Decorate(value),
        Command::Expr(expr) => DebugCommand::Expr(debug_expr_id(context, *expr)),
    }
}

fn debug_expr_id<'a>(context: &'a Context, expr: ExprId) -> DebugExpr<'a> {
    debug_expr(context, &context.exprs[expr])
}

fn debug_expr_list<'a>(context: &'a Context, exprs: &[ExprId]) -> Vec<DebugExpr<'a>> {
    exprs
        .iter()
        .map(|expr| debug_expr_id(context, *expr))
        .collect()
}

fn debug_expr<'a>(context: &'a Context, expr: &'a Expr) -> DebugExpr<'a> {
    match expr {
        Expr::Cast { expr, ty } => DebugExpr::Cast {
            expr: Box::new(debug_expr_id(context, *expr)),
            ty: debug_type_id(context, *ty),
        },
        Expr::ArrayLiteral(elements) => DebugExpr::ArrayLiteral(debug_expr_list(
            context,
            elements.as_slice(&context.expr_lists),
        )),
        Expr::RecordLiteral(fields) => DebugExpr::RecordLiteral(
            sorted_entries(fields)
                .into_iter()
                .map(|(id, expr)| (id, debug_expr_id(context, *expr)))
                .collect(),
        ),
        Expr::RationalLiteral(value) => DebugExpr::RationalLiteral(value),
        Expr::IntLiteral { value, base } => DebugExpr::IntLiteral {
            value: *value,
            base: *base,
        },
        Expr::BoolLiteral(value) => DebugExpr::BoolLiteral(*value),
        Expr::ArrayAccess { array, indices } => DebugExpr::ArrayAccess {
            array,
            indices: debug_expr_list(context, indices.as_slice(&context.expr_lists)),
        },
        Expr::RecordAccess { record, field } => DebugExpr::RecordAccess {
            record: Box::new(debug_expr_id(context, *record)),
            field,
        },
        Expr::Application { func, args } => DebugExpr::Application {
            func,
            args: debug_expr_list(context, args.as_slice(&context.expr_lists)),
        },
        Expr::Id(id) => DebugExpr::Id(id),
        Expr::BinOp { left, op, right } => DebugExpr::BinOp {
            left: Box::new(debug_expr_id(context, *left)),
            op,
            right: Box::new(debug_expr_id(context, *right)),
        },
    }
}

fn debug_type_id<'a>(context: &'a Context, ty: TypeId) -> DebugType<'a> {
    debug_type(context, &context.types[ty])
}

fn debug_type_list<'a>(context: &'a Context, types: &[TypeId]) -> Vec<DebugType<'a>> {
    types.iter().map(|ty| debug_type_id(context, *ty)).collect()
}

fn debug_type<'a>(context: &'a Context, ty: &'a Type) -> DebugType<'a> {
    match ty {
        Type::Float => DebugType::Float,
        Type::Double => DebugType::Double,
        Type::Bool => DebugType::Bool,
        Type::Bit { length, unsigned } => DebugType::Bit {
            length: *length,
            unsigned: *unsigned,
        },
        Type::Fixed {
            length_total,
            length_int,
            unsigned,
        } => DebugType::Fixed {
            length_total: *length_total,
            length_int: *length_int,
            unsigned: *unsigned,
        },
        Type::Alias(name) => DebugType::Alias(name),
        Type::Array {
            element_type,
            dims,
            ports,
        } => DebugType::Array {
            element_type: Box::new(debug_type_id(context, *element_type)),
            dims,
            ports: *ports,
        },
        Type::StaticInt(value) => DebugType::StaticInt(*value),
        Type::Index { static_, dynamic } => DebugType::Index {
            static_: *static_,
            dynamic: *dynamic,
        },
        Type::Void => DebugType::Void,
        Type::Rational(value) => DebugType::Rational(value),
        Type::Func { args, ret } => DebugType::Func {
            args: debug_type_list(context, args.as_slice(&context.type_lists)),
            ret: Box::new(debug_type_id(context, *ret)),
        },
        Type::RecType { name, fields } => DebugType::RecType {
            name,
            fields: sorted_entries(fields)
                .into_iter()
                .map(|(id, ty)| (id, debug_type_id(context, *ty)))
                .collect(),
        },
    }
}

fn sorted_entries<V>(fields: &std::collections::HashMap<Id, V>) -> Vec<(&Id, &V)> {
    let mut fields: Vec<_> = fields.iter().collect();
    fields.sort_by(|(left, _), (right, _)| left.0.cmp(&right.0));
    fields
}
