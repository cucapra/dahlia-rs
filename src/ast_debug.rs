use std::collections::HashMap;

use serde::{Serialize, Serializer};
use serde_json::{Map, Value, json};

use crate::ast::{
    AssignOp, Backend, Command, CommandId, Context, Decl, Def, DimSpec, Expr, ExprId, FieldId,
    ForRange, FuncId, IdResolve, Include, InfixOp, Program, RecordId, Suffix, Type, TypeId,
    ValueId, View,
};

pub struct DebugAst<'a> {
    context: &'a Context,
    program: &'a Program,
}

impl Serialize for DebugAst<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        debug_program(self.context, self.program).serialize(serializer)
    }
}

pub fn debug_ast<'a>(context: &'a Context, program: &'a Program) -> DebugAst<'a> {
    DebugAst { context, program }
}

pub fn ast_debug_json(context: &Context, program: &Program) -> String {
    serde_json::to_string_pretty(&debug_ast(context, program)).expect("Debug AST should serialize")
}

pub fn ast_debug(context: &Context, program: &Program) {
    println!("{}", ast_debug_json(context, program));
}

fn type_annotation(context: &Context, ty: Option<TypeId>) -> Value {
    ty.map(|ty| debug_type_id(context, ty))
        .unwrap_or(Value::Null)
}

fn debug_field_id(context: &Context, id: FieldId) -> Value {
    json!({
        "kind": "FieldId",
        "v": id.resolve_id(&context.ast),
        "typ": type_annotation(context, None),
    })
}

fn debug_record_id(context: &Context, id: RecordId) -> Value {
    json!({
        "kind": "RecordId",
        "v": id.resolve_id(&context.ast),
        "typ": type_annotation(context, None),
    })
}

fn debug_value_id(context: &Context, id: ValueId) -> Value {
    json!({
        "kind": "ValueId",
        "v": id.resolve_id(&context.ast),
        "typ": type_annotation(context, context.tcx.value_type_map.get(&id).copied()),
    })
}

fn debug_func_id(context: &Context, id: FuncId) -> Value {
    json!({
        "kind": "FuncId",
        "v": id.resolve_id(&context.ast),
        "typ": type_annotation(context, context.tcx.func_type_map.get(&id).copied()),
    })
}

fn debug_program(context: &Context, program: &Program) -> Value {
    json!({
        "kind": "Prog",
        "includes": program.includes.iter().map(|include| debug_include(context, include)).collect::<Vec<_>>(),
        "defs": program.defs.iter().map(|def| debug_def(context, def, true)).collect::<Vec<_>>(),
        "decors": debug_command_list(context, program.decors.as_slice(&context.ast.command_lists)),
        "decls": program.decls.iter().map(|decl| debug_decl(context, decl)).collect::<Vec<_>>(),
        "cmd": debug_command_id(context, program.cmd),
    })
}

fn debug_include(context: &Context, include: &Include) -> Value {
    let backends = Map::from_iter(
        include
            .backends
            .iter()
            .map(|(backend, value)| (backend_name(backend).to_string(), json!(value))),
    );

    json!({
        "kind": "Include",
        "backends": Value::Object(backends),
        "defs": include.defs.iter().map(|def| debug_def(context, def, false)).collect::<Vec<_>>(),
    })
}

fn debug_def(context: &Context, def: &Def, include_body: bool) -> Value {
    match def {
        Def::Func { sig, body } => json!({
            "kind": "FuncDef",
            "id": debug_func_id(context, sig.name),
            "args": debug_decl_list(context, &sig.args),
            "retTy": debug_type_id(context, sig.ret_ty),
            "body": if include_body { debug_command_id(context, *body) } else { Value::Null },
        }),
        Def::Record { name, fields } => json!({
            "kind": "RecordDef",
            "name": debug_record_id(context, *name),
            "fields": debug_field_entries(context, fields),
        }),
    }
}

fn debug_decl_list(context: &Context, decls: &[Decl]) -> Vec<Value> {
    decls.iter().map(|decl| debug_decl(context, decl)).collect()
}

fn debug_decl(context: &Context, decl: &Decl) -> Value {
    json!({
        "kind": "Decl",
        "id": debug_value_id(context, decl.id),
        "typ": debug_type_id(context, decl.ty),
    })
}

fn debug_field_entries(context: &Context, fields: &HashMap<FieldId, TypeId>) -> Vec<Value> {
    sorted_entries(context, fields)
        .into_iter()
        .map(|(id, ty)| json!({ "name": debug_field_id(context, id), "value": debug_type_id(context, *ty) }))
        .collect()
}

fn debug_expr_field_entries(context: &Context, fields: &HashMap<FieldId, ExprId>) -> Vec<Value> {
    sorted_entries(context, fields)
        .into_iter()
        .map(|(id, expr)| json!({ "name": debug_field_id(context, id), "value": debug_expr_id(context, *expr) }))
        .collect()
}

fn debug_view(context: &Context, view: &View) -> Value {
    json!({
        "kind": "View",
        "suffix": debug_suffix(context, &view.suffix),
        "prefix": view.prefix,
        "shrink": view.shrink,
    })
}

fn debug_suffix(context: &Context, suffix: &Suffix) -> Value {
    match suffix {
        Suffix::Rotation(expr) => json!({
            "kind": "Rotation",
            "e": debug_expr_id(context, *expr),
        }),
        Suffix::Aligned { factor, e } => json!({
            "kind": "Aligned",
            "factor": factor,
            "e": debug_expr_id(context, *e),
        }),
    }
}

fn debug_for_range(context: &Context, range: &ForRange) -> Value {
    json!({
        "kind": "CRange",
        "iter": debug_value_id(context, range.iter),
        "castType": type_annotation(context, range.ty),
        "reversed": range.rev,
        "s": range.start,
        "e": range.end,
        "u": range.unroll,
    })
}

fn debug_command_id(context: &Context, command: CommandId) -> Value {
    debug_command(context, &context.ast.commands[command])
}

fn debug_command_list(context: &Context, commands: &[CommandId]) -> Vec<Value> {
    commands
        .iter()
        .map(|command| debug_command_id(context, *command))
        .collect()
}

fn debug_command(context: &Context, command: &Command) -> Value {
    match command {
        Command::Empty => json!({
            "kind": "CEmpty",
        }),
        Command::Par(commands) => json!({
            "kind": "CPar",
            "cmds": debug_command_list(context, commands),
        }),
        Command::Seq(commands) => json!({
            "kind": "CSeq",
            "cmds": debug_command_list(context, commands),
        }),
        Command::Let { id, ty, value } => json!({
            "kind": "CLet",
            "id": debug_value_id(context, *id),
            "typ": type_annotation(context, *ty),
            "e": value.map(|expr| debug_expr_id(context, expr)).unwrap_or(Value::Null),
        }),
        Command::Update { lhs, op, rhs } => match op {
            AssignOp::Assign => json!({
                "kind": "CUpdate",
                "lhs": debug_expr_id(context, *lhs),
                "rhs": debug_expr_id(context, *rhs),
            }),
            AssignOp::AddAssign
            | AssignOp::SubAssign
            | AssignOp::MulAssign
            | AssignOp::DivAssign => json!({
                "kind": "CReduce",
                "rop": debug_rop(op),
                "lhs": debug_expr_id(context, *lhs),
                "rhs": debug_expr_id(context, *rhs),
            }),
        },
        Command::View { id, arr_id, dims } => json!({
            "kind": "CView",
            "id": debug_value_id(context, *id),
            "arrId": debug_value_id(context, *arr_id),
            "dims": dims.iter().map(|view| debug_view(context, view)).collect::<Vec<_>>(),
        }),
        Command::Split { id, arr_id, dims } => json!({
            "kind": "CSplit",
            "id": debug_value_id(context, *id),
            "arrId": debug_value_id(context, *arr_id),
            "factors": dims,
        }),
        Command::Return(expr) => json!({
            "kind": "CReturn",
            "exp": debug_expr_id(context, *expr),
        }),
        Command::IfElse { cond, then, else_ } => json!({
            "kind": "CIf",
            "cond": debug_expr_id(context, *cond),
            "cons": debug_command_id(context, *then),
            "alt": debug_command_id(context, *else_),
        }),
        Command::While {
            cond,
            pipeline,
            body,
        } => json!({
            "kind": "CWhile",
            "cond": debug_expr_id(context, *cond),
            "pipeline": pipeline,
            "body": debug_command_id(context, *body),
        }),
        Command::For {
            range,
            pipeline,
            body,
            combine,
        } => json!({
            "kind": "CFor",
            "range": debug_for_range(context, range),
            "pipeline": pipeline,
            "par": debug_command_id(context, *body),
            "combine": debug_command_id(context, *combine),
        }),
        Command::Decorate(value) => json!({
            "kind": "CDecorate",
            "value": value,
        }),
        Command::Expr(expr) => json!({
            "kind": "CExpr",
            "exp": debug_expr_id(context, *expr),
        }),
    }
}

fn debug_expr_id(context: &Context, expr: ExprId) -> Value {
    let mut value = debug_expr(context, expr, &context.ast.exprs[expr]);
    value["typ"] = type_annotation(context, context.tcx.expr_type_map.get(expr).copied());
    value
}

fn debug_expr_list(context: &Context, exprs: &[ExprId]) -> Vec<Value> {
    exprs
        .iter()
        .map(|expr| debug_expr_id(context, *expr))
        .collect()
}

fn debug_expr(context: &Context, _expr_id: ExprId, expr: &Expr) -> Value {
    match expr {
        Expr::Cast { expr, ty } => json!({
            "kind": "ECast",
            "e": debug_expr_id(context, *expr),
            "castType": debug_type_id(context, *ty),
        }),
        Expr::ArrayLiteral(elements) => json!({
            "kind": "EArrLiteral",
            "idxs": debug_expr_list(context, elements.as_slice(&context.ast.expr_lists)),
        }),
        Expr::RecordLiteral(fields) => json!({
            "kind": "ERecLiteral",
            "fields": debug_expr_field_entries(context, fields),
        }),
        Expr::RationalLiteral(value) => json!({
            "kind": "ERational",
            "d": value,
        }),
        Expr::IntLiteral { value, base } => json!({
            "kind": "EInt",
            "v": value.to_string(),
            "base": base,
        }),
        Expr::BoolLiteral(value) => json!({
            "kind": "EBool",
            "v": value,
        }),
        Expr::ArrayAccess { array, indices } => json!({
            "kind": "EArrAccess",
            "id": debug_value_id(context, *array),
            "idxs": debug_expr_list(context, indices.as_slice(&context.ast.expr_lists)),
        }),
        Expr::RecordAccess { record, field } => json!({
            "kind": "ERecAccess",
            "rec": debug_expr_id(context, *record),
            "fieldName": debug_field_id(context, *field),
        }),
        Expr::Application { func, args } => json!({
            "kind": "EApp",
            "func": debug_func_id(context, *func),
            "args": debug_expr_list(context, args.as_slice(&context.ast.expr_lists)),
        }),
        Expr::Id(id) => json!({
            "kind": "EVar",
            "id": debug_value_id(context, *id),
        }),
        Expr::BinOp { left, op, right } => json!({
            "kind": "EBinop",
            "op": debug_bop(op),
            "e1": debug_expr_id(context, *left),
            "e2": debug_expr_id(context, *right),
        }),
    }
}

fn debug_type_id(context: &Context, ty: TypeId) -> Value {
    debug_type(context, &context.tcx.types[ty])
}

fn debug_type_list(context: &Context, types: &[TypeId]) -> Vec<Value> {
    types.iter().map(|ty| debug_type_id(context, *ty)).collect()
}

fn debug_type(context: &Context, ty: &Type) -> Value {
    match ty {
        Type::Float => json!({ "kind": "TFloat" }),
        Type::Double => json!({ "kind": "TDouble" }),
        Type::Bool => json!({ "kind": "TBool" }),
        Type::Bit { length, unsigned } => json!({
            "kind": "TSizedInt",
            "len": length,
            "unsigned": unsigned,
        }),
        Type::Fixed {
            length_total,
            length_int,
            unsigned,
        } => json!({
            "kind": "TFixed",
            "ltotal": length_total,
            "lint": length_int,
            "unsigned": unsigned,
        }),
        Type::Alias(name) => json!({
            "kind": "TAlias",
            "name": debug_record_id(context, *name),
        }),
        Type::Array {
            element_type,
            dims,
            ports,
        } => json!({
            "kind": "TArray",
            "typ": debug_type_id(context, *element_type),
            "dims": dims.iter().map(debug_dim_spec).collect::<Vec<_>>(),
            "ports": ports,
        }),
        Type::StaticInt(value) => json!({
            "kind": "TStaticInt",
            "v": value.to_string(),
        }),
        Type::Index { static_, dynamic } => json!({
            "kind": "TIndex",
            "static": { "lo": static_.0, "hi": static_.1 },
            "dynamic": { "lo": dynamic.0, "hi": dynamic.1 },
        }),
        Type::Void => json!({ "kind": "TVoid" }),
        Type::Rational(value) => json!({
            "kind": "TRational",
            "value": value,
        }),
        Type::Func { args, ret } => json!({
            "kind": "TFun",
            "args": debug_type_list(context, args.as_slice(&context.tcx.type_lists)),
            "ret": debug_type_id(context, *ret),
        }),
        Type::RecType { name, fields } => json!({
            "kind": "TRecType",
            "name": debug_record_id(context, *name),
            "fields": debug_field_entries(context, fields),
        }),
    }
}

fn debug_dim_spec(dim: &DimSpec) -> Value {
    json!({
        "len": dim.length,
        "bank": dim.bank,
    })
}

fn debug_bop(op: &InfixOp) -> Value {
    let (kind, op) = match op {
        InfixOp::Mul => ("NumOp", "*"),
        InfixOp::Div => ("NumOp", "/"),
        InfixOp::Mod => ("NumOp", "%"),
        InfixOp::Add => ("NumOp", "+"),
        InfixOp::Sub => ("NumOp", "-"),
        InfixOp::Shl => ("BitOp", "<<"),
        InfixOp::Shr => ("BitOp", ">>"),
        InfixOp::Eq => ("EqOp", "=="),
        InfixOp::Neq => ("EqOp", "!="),
        InfixOp::Le => ("CmpOp", "<="),
        InfixOp::Ge => ("CmpOp", ">="),
        InfixOp::Lt => ("CmpOp", "<"),
        InfixOp::Gt => ("CmpOp", ">"),
        InfixOp::And => ("BoolOp", "&&"),
        InfixOp::Or => ("BoolOp", "||"),
        InfixOp::Band => ("BitOp", "&"),
        InfixOp::Bor => ("BitOp", "|"),
        InfixOp::Bxor => ("BitOp", "^"),
    };

    json!({
        "kind": kind,
        "op": op,
    })
}

fn debug_rop(op: &AssignOp) -> Value {
    let op = match op {
        AssignOp::Assign => ":=",
        AssignOp::AddAssign => "+=",
        AssignOp::SubAssign => "-=",
        AssignOp::MulAssign => "*=",
        AssignOp::DivAssign => "/=",
    };

    json!({
        "kind": "ROp",
        "op": op,
    })
}

fn backend_name(backend: &Backend) -> &'static str {
    match backend {
        Backend::Cpp => "c++",
        Backend::Vivado => "vivado",
        Backend::Futil | Backend::Calyx => "calyx",
    }
}

fn sorted_entries<'a, V>(
    context: &'a Context,
    fields: &'a HashMap<FieldId, V>,
) -> Vec<(FieldId, &'a V)>
where
    V: Ord,
{
    let mut fields: Vec<_> = fields.iter().map(|(id, v)| (*id, v)).collect();
    fields.sort_by(|(left, left_v), (right, right_v)| {
        left.resolve_id(&context.ast)
            .cmp(right.resolve_id(&context.ast))
            .then_with(|| left_v.cmp(right_v))
    });
    fields
}
