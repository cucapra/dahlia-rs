use pest::pratt_parser::PrattParser;
use pest_consume::{Error, Parser, match_nodes};

use crate::ast::{
    AssignOp, Backend, Command, Decl, Def, DimSpec, Expr, ForRange, FuncSig, Id, Include, InfixOp,
    Program, Suffix, Type, TypeAtom, View,
};

type Result<T> = std::result::Result<T, Error<Rule>>;
type Node<'i> = pest_consume::Node<'i, Rule, ()>;

lazy_static::lazy_static! {
    static ref PRATT_PARSER: PrattParser<Rule> = {
        use pest::pratt_parser::{Assoc::*, Op};
        use Rule::*;

        // Precedence is defined lowest to highest
        PrattParser::new()
            // Addition and subtract have equal precedence
            .op(Op::infix(or, Left))
            .op(Op::infix(and, Left))
            .op(Op::infix(bor, Left))
            .op(Op::infix(bxor, Left))
            .op(Op::infix(band, Left))
            .op(Op::infix(shr, Left) | Op::infix(shl, Left))
            .op(Op::infix(eq, Left) | Op::infix(neq, Left) | Op::infix(ge, Left) | Op::infix(le, Left) | Op::infix(gt, Left) | Op::infix(lt, Left))
            .op(Op::infix(add, Left) | Op::infix(sub, Left))
            .op(Op::infix(mul, Left) | Op::infix(div, Left) | Op::infix(modulo, Left))
    };
}

#[derive(pest_consume::Parser)]
#[grammar = "dahlia.pest"]
pub struct DahliaParser;

#[pest_consume::parser]
impl DahliaParser {
    fn EOI(_input: Node) -> Result<()> {
        Ok(())
    }

    fn number(input: Node) -> Result<usize> {
        Ok(input.as_str().parse().unwrap())
    }

    fn iden(input: Node) -> Result<Id> {
        Ok(Id(input.as_str().to_string()))
    }

    fn ty_float(_input: Node) -> Result<TypeAtom> {
        Ok(TypeAtom::Float)
    }

    fn ty_double(_input: Node) -> Result<TypeAtom> {
        Ok(TypeAtom::Double)
    }

    fn ty_bool(_input: Node) -> Result<TypeAtom> {
        Ok(TypeAtom::Bool)
    }

    fn ty_bit(input: Node) -> Result<TypeAtom> {
        Ok(match_nodes!(input.into_children();
            [number(length)] => TypeAtom::Bit{length, unsigned: false}
        ))
    }

    fn ty_ubit(input: Node) -> Result<TypeAtom> {
        Ok(match_nodes!(input.into_children();
            [number(length)] => TypeAtom::Bit{length, unsigned: true}
        ))
    }

    fn ty_fix(input: Node) -> Result<TypeAtom> {
        Ok(match_nodes!(input.into_children();
            [number(length_total), number(length_int)] => TypeAtom::Fixed{length_total, length_int, unsigned: false}
        ))
    }

    fn ty_ufix(input: Node) -> Result<TypeAtom> {
        Ok(match_nodes!(input.into_children();
            [number(length_total), number(length_int)] => TypeAtom::Fixed{length_total, length_int, unsigned: true}
        ))
    }

    fn ty_atom(input: Node) -> Result<TypeAtom> {
        match_nodes!(input.into_children();
            [ty_float(t)] => Ok(t),
            [ty_double(t)] => Ok(t),
            [ty_bool(t)] => Ok(t),
            [ty_bit(t)] => Ok(t),
            [ty_ubit(t)] => Ok(t),
            [ty_fix(t)] => Ok(t),
            [ty_ufix(t)] => Ok(t),
            [iden(id)] => Ok(TypeAtom::Alias(id))
        )
    }

    fn ty_idx(input: Node) -> Result<DimSpec> {
        Ok(match_nodes!(input.into_children();
            [number(length)] => DimSpec{length, bank: None},
            [number(length), number(bank)] => DimSpec{length, bank: Some(bank)}
        ))
    }

    fn ty(input: Node) -> Result<Type> {
        Ok(match_nodes!(input.into_children();
            [ty_atom(t)] => Type::Simple(t),
            [ty_atom(element_type), ty_idx(idx)..] => Type::Array{element_type, ports: 1, dims: idx.into_iter().collect()},
            [ty_atom(element_type), number(ports), ty_idx(idx)..] => Type::Array{element_type, ports, dims: idx.into_iter().collect()},
        ))
    }

    fn expr_cast(input: Node) -> Result<Expr> {
        Ok(match_nodes!(input.into_children();
            [expr(e), ty_atom(ty)] => Expr::Cast{expr: Box::new(e), ty}
        ))
    }

    fn rec_lit_field(input: Node) -> Result<(Id, Expr)> {
        Ok(match_nodes!(input.into_children();
            [iden(id), expr(e)] => (id, e)
        ))
    }

    fn compound_literal(input: Node) -> Result<Expr> {
        Ok(match_nodes!(input.into_children();
            [rec_lit_field(fields)..] => Expr::RecordLiteral(fields.into_iter().collect()),
            [expr(elements)..] => Expr::ArrayLiteral(elements.into_iter().collect())
        ))
    }

    fn array_access(input: Node) -> Result<Expr> {
        Ok(match_nodes!(input.into_children();
            [iden(array), expr(indices)..] => Expr::ArrayAccess{array, indices: indices.into_iter().collect()}
        ))
    }

    fn rational(input: Node) -> Result<Expr> {
        Ok(Expr::RationalLiteral(input.as_str().to_string()))
    }

    fn hex(input: Node) -> Result<Expr> {
        Ok(Expr::IntLiteral {
            value: i64::from_str_radix(input.as_str().trim_start_matches("0x"), 16)
                .expect("Invalid hex literal"),
            base: 16,
        })
    }

    fn octal(input: Node) -> Result<Expr> {
        Ok(Expr::IntLiteral {
            value: i64::from_str_radix(input.as_str().trim_start_matches("0"), 8)
                .expect("Invalid octal literal"),
            base: 8,
        })
    }

    fn uint(input: Node) -> Result<Expr> {
        Ok(Expr::IntLiteral {
            value: input.as_str().parse().expect("Invalid integer literal"),
            base: 10,
        })
    }

    fn boolean(input: Node) -> Result<Expr> {
        match input.as_str() {
            "true" => Ok(Expr::BoolLiteral(true)),
            "false" => Ok(Expr::BoolLiteral(false)),
            _ => unreachable!(),
        }
    }

    fn app(input: Node) -> Result<Expr> {
        Ok(match_nodes!(input.into_children();
            [iden(func)] => Expr::Application{func, args: vec![]},
            [iden(func), expr(args)..] => Expr::Application{func, args: args.into_iter().collect()}
        ))
    }

    fn primary(input: Node) -> Result<Expr> {
        Ok(match_nodes!(input.into_children();
            [expr_cast(e)] => e,
            [compound_literal(e)] => e,
            [rational(e)] => e,
            [hex(e)] => e,
            [octal(e)] => e,
            [uint(e)] => e,
            [boolean(e)] => e,
            [array_access(e)] => e,
            [app(e)] => e,
            [iden(id)] => Expr::Id(id),
            [expr(e)] => e
        ))
    }

    fn atom(input: Node) -> Result<Expr> {
        match_nodes!(input.into_children();
            [primary(e)] => Ok(e),
            [primary(e), iden(fields)..] => Ok(
                fields.into_iter().fold(e, |acc, field| Expr::RecordAccess{record: Box::new(acc), field})
            )
        )
    }

    fn expr(input: Node) -> Result<Expr> {
        PRATT_PARSER
            .map_primary(|primary| match primary.as_rule() {
                Rule::atom => Self::atom(Node::new(primary)),
                _ => unreachable!("Unexpected primary expression: {:?}", primary.as_rule()),
            })
            .map_infix(|lhs, op, rhs| {
                let op = match op.as_rule() {
                    Rule::mul => InfixOp::Mul,
                    Rule::div => InfixOp::Div,
                    Rule::modulo => InfixOp::Mod,
                    Rule::add => InfixOp::Add,
                    Rule::sub => InfixOp::Sub,
                    Rule::shl => InfixOp::Shl,
                    Rule::shr => InfixOp::Shr,
                    Rule::eq => InfixOp::Eq,
                    Rule::neq => InfixOp::Neq,
                    Rule::le => InfixOp::Le,
                    Rule::ge => InfixOp::Ge,
                    Rule::lt => InfixOp::Lt,
                    Rule::gt => InfixOp::Gt,
                    Rule::and => InfixOp::And,
                    Rule::or => InfixOp::Or,
                    Rule::band => InfixOp::Band,
                    Rule::bor => InfixOp::Bor,
                    Rule::bxor => InfixOp::Bxor,
                    _ => unreachable!("Unexpected infix operator: {:?}", op.as_rule()),
                };
                Ok(Expr::BinOp {
                    left: Box::new(lhs?),
                    op,
                    right: Box::new(rhs?),
                })
            })
            .parse(input.into_pair().into_inner())
    }

    fn let_stmt(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [iden(id), ty(ty), expr(value)] => Command::Let{id, ty: Some(ty), value: Some(value)},
            [iden(id), ty(ty)] => Command::Let{id, ty: Some(ty), value: None},
            [iden(id), expr(value)] => Command::Let{id, ty: None, value: Some(value)},
            [iden(id)] => Command::Let{id, ty: None, value: None}
        ))
    }

    fn assign(_input: Node) -> Result<AssignOp> {
        Ok(AssignOp::Assign)
    }

    fn add_assign(_input: Node) -> Result<AssignOp> {
        Ok(AssignOp::AddAssign)
    }

    fn sub_assign(_input: Node) -> Result<AssignOp> {
        Ok(AssignOp::SubAssign)
    }

    fn mul_assign(_input: Node) -> Result<AssignOp> {
        Ok(AssignOp::MulAssign)
    }

    fn div_assign(_input: Node) -> Result<AssignOp> {
        Ok(AssignOp::DivAssign)
    }

    fn assign_op(input: Node) -> Result<AssignOp> {
        Ok(match_nodes!(input.into_children();
            [assign(op)] => op,
            [add_assign(op)] => op,
            [sub_assign(op)] => op,
            [mul_assign(op)] => op,
            [div_assign(op)] => op
        ))
    }

    fn update(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [expr(lhs), assign_op(op), expr(rhs)] => Command::Update{lhs, op, rhs}
        ))
    }

    fn view_suffix_underscore(_input: Node) -> Result<()> {
        Ok(())
    }

    fn view_suffix(input: Node) -> Result<Suffix> {
        Ok(match_nodes!(input.into_children();
            [view_suffix_underscore(_)] => Suffix::Rotation(Expr::IntLiteral{value:0, base:10}),
            [number(factor), expr(e)] => Suffix::Aligned { factor, e },
            [expr(e)] => Suffix::Rotation(e)
        ))
    }

    fn view_prefix_opt(input: Node) -> Result<Option<usize>> {
        Ok(match_nodes!(input.into_children();
            [number(n)] => Some(n),
            [] => None
        ))
    }

    fn view_shrink_opt(input: Node) -> Result<Option<usize>> {
        Ok(match_nodes!(input.into_children();
            [number(n)] => Some(n),
            [] => None
        ))
    }

    fn view_param(input: Node) -> Result<View> {
        Ok(match_nodes!(input.into_children();
            [view_suffix(suffix), view_prefix_opt(prefix), view_shrink_opt(shrink)] => View{prefix, suffix, shrink}
        ))
    }

    fn view(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [iden(id), iden(arr_id), view_param(dims)..] => Command::View{id, arr_id, dims: dims.into_iter().collect()}
        ))
    }

    fn split_factor(input: Node) -> Result<usize> {
        Ok(match_nodes!(input.into_children();
            [number(factor)] => factor
        ))
    }

    fn split(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [iden(id), iden(arr_id), split_factor(dims)..] => Command::Split{id, arr_id, dims: dims.into_iter().collect()}
        ))
    }

    fn r#return(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [expr(e)] => Command::Return(e),
        ))
    }

    fn simple_cmd(input: Node) -> Result<Command> {
        match_nodes!(input.into_children();
            [let_stmt(cmd)] => Ok(cmd),
            [update(cmd)] => Ok(cmd),
            [view(cmd)] => Ok(cmd),
            [r#return(cmd)] => Ok(cmd),
            [split(cmd)] => Ok(cmd),
            [expr(e)] => Ok(Command::Expr(e))
        )
    }

    fn cmd(input: Node) -> Result<Command> {
        match_nodes!(input.into_children();
            [par_cmd(c)..] => Ok(Command::Par(c.into_iter().collect())),
        )
    }

    fn block(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [cmd(c)] => c,
        ))
    }

    fn else_block(input: Node) -> Result<Option<Command>> {
        Ok(match_nodes!(input.into_children();
            [block(cmd)] => Some(cmd),
            [] => None
        ))
    }

    fn if_else(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [expr(cond), block(then), else_block(else_)] => Command::IfElse{cond, then:Box::new(then), else_: else_.map(Box::new)}
        ))
    }

    fn kw_pipeline(_input: Node) -> Result<()> {
        Ok(())
    }

    fn pipeline(input: Node) -> Result<bool> {
        Ok(match_nodes!(input.into_children();
            [kw_pipeline(_)] => true,
            [] => false
        ))
    }

    fn kw_for_rev(_input: Node) -> Result<()> {
        Ok(())
    }

    fn for_rev(input: Node) -> Result<bool> {
        Ok(match_nodes!(input.into_children();
            [kw_for_rev(_)] => true,
            [] => false
        ))
    }

    fn while_loop(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [expr(cond), block(body)] => Command::While{cond, pipeline:false, body: Box::new(body)},
            [expr(cond), pipeline(_), block(body)] => Command::While{cond, pipeline:true, body: Box::new(body)}
        ))
    }

    fn for_range(input: Node) -> Result<ForRange> {
        Ok(match_nodes!(input.into_children();
            [iden(id), ty(ty), for_rev(rev), number(start), number(end), number(unroll)] => ForRange{id, ty: Some(ty), rev, start, end, unroll},
            [iden(id), ty(ty), for_rev(rev), number(start), number(end), ] => ForRange{id, ty: Some(ty), rev, start, end, unroll:1},
            [iden(id), for_rev(rev), number(start), number(end), number(unroll)] => ForRange{id, ty: None, rev, start, end, unroll},
            [iden(id), for_rev(rev), number(start), number(end)] => ForRange{id, ty: None, rev, start, end, unroll:1},
        ))
    }

    fn combine_block(input: Node) -> Result<Option<Command>> {
        Ok(match_nodes!(input.into_children();
            [block(cmd)] => Some(cmd),
            [] => None
        ))
    }

    fn string_val(input: Node) -> Result<String> {
        Ok(input.as_str().trim_matches('"').to_string())
    }

    fn decor(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [string_val(value)] => Command::Decorate(value)
        ))
    }

    fn decors(input: Node) -> Result<Vec<Command>> {
        Ok(match_nodes!(input.into_children();
            [decor(decs)..] => decs.into_iter().collect()
        ))
    }

    fn for_loop(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [for_range(range), pipeline(pipeline), block(body), combine_block(combine)] => Command::For{range, pipeline, body: Box::new(body), combine: combine.map(Box::new)},
        ))
    }

    fn par_cmd_item(input: Node) -> Result<Command> {
        match_nodes!(input.into_children();
            [block_cmd(c)] => Ok(c),
            [simple_cmd(c)] => Ok(c)
        )
    }

    fn par_cmd(input: Node) -> Result<Command> {
        match_nodes!(input.into_children();
            [par_cmd_item(c)..] => Ok(Command::Par(c.into_iter().collect()))
        )
    }

    fn arg(input: Node) -> Result<Decl> {
        Ok(match_nodes!(input.into_children();
            [iden(id), ty(ty)] => Decl{id, ty}
        ))
    }

    fn func_args(input: Node) -> Result<Vec<Decl>> {
        Ok(match_nodes!(input.into_children();
            [arg(arg)..] => arg.into_iter().collect()
        ))
    }

    fn func_def(input: Node) -> Result<Def> {
        Ok(match_nodes!(input.into_children();
            [iden(name), func_args(args), ty(ret_ty), block(body)] => Def::Func{sig: FuncSig{name, args, ret_ty: Some(ret_ty)}, body},
            [iden(name), func_args(args), block(body)] => Def::Func{sig: FuncSig{name, args, ret_ty: None}, body}
        ))
    }

    fn rec_field_defs(input: Node) -> Result<Vec<Decl>> {
        Ok(match_nodes!(input.into_children();
            [arg(arg)..] => arg.into_iter().collect()
        ))
    }

    fn record_def(input: Node) -> Result<Def> {
        Ok(match_nodes!(input.into_children();
            [iden(name), rec_field_defs(fields)] => Def::Record{ name, fields }
        ))
    }

    fn def(input: Node) -> Result<Def> {
        match_nodes!(input.into_children();
            [func_def(def)] => Ok(def),
            [record_def(def)] => Ok(def)
        )
    }

    fn defs(input: Node) -> Result<Vec<Def>> {
        Ok(match_nodes!(input.into_children();
            [def(defs)..] => defs.into_iter().collect()
        ))
    }

    fn decls(input: Node) -> Result<Vec<Decl>> {
        Ok(match_nodes!(input.into_children();
            [arg(decls)..] => decls.into_iter().collect()
        ))
    }

    fn block_cmd(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [block(c)] => c,
            [if_else(c)] => c,
            [while_loop(c)] => c,
            [for_loop(c)] => c,
            [decor(c)] => c,
        ))
    }

    fn prog_cmd(input: Node) -> Result<Option<Command>> {
        Ok(match_nodes!(input.into_children();
            [cmd(cmd)] => Some(cmd),
            [] => None
        ))
    }

    fn backend_cpp(_input: Node) -> Result<Backend> {
        Ok(Backend::Cpp)
    }

    fn backend_vivado(_input: Node) -> Result<Backend> {
        Ok(Backend::Vivado)
    }

    fn backend_futil(_input: Node) -> Result<Backend> {
        Ok(Backend::Futil)
    }

    fn backend_calyx(_input: Node) -> Result<Backend> {
        Ok(Backend::Calyx)
    }

    fn backend(input: Node) -> Result<Backend> {
        Ok(match_nodes!(input.into_children();
            [backend_cpp(b)] => b,
            [backend_vivado(b)] => b,
            [backend_futil(b)] => b,
            [backend_calyx(b)] => b))
    }

    fn func_signature(input: Node) -> Result<FuncSig> {
        Ok(match_nodes!(input.into_children();
            [iden(name), func_args(args), ty(ret_ty)] => FuncSig{name, args, ret_ty: Some(ret_ty)},
            [iden(name), func_args(args)] => FuncSig{name, args, ret_ty: None}
        ))
    }

    fn func_signatures(input: Node) -> Result<Vec<FuncSig>> {
        Ok(match_nodes!(input.into_children();
            [func_signature(sigs)..] => sigs.into_iter().collect()
        ))
    }

    fn backend_opt(input: Node) -> Result<(Backend, String)> {
        Ok(match_nodes!(input.into_children();
            [backend(b), string_val(s)] => (b, s)
        ))
    }

    fn include(input: Node) -> Result<Include> {
        Ok(match_nodes!(input.into_children();
            [backend_opt(backends).., func_signatures(sigs)] => Include{ backends: backends.into_iter().collect(), defs: sigs }
        ))
    }

    fn includes(input: Node) -> Result<Vec<Include>> {
        Ok(match_nodes!(input.into_children();
            [include(includes)..] => includes.into_iter().collect()
        ))
    }

    fn prog(input: Node) -> Result<Program> {
        Ok(match_nodes!(input.into_children();
            [includes(includes), defs(defs), decors(decors), decls(decls), prog_cmd(cmd), EOI(_)] => Program{includes, defs, decors, decls, cmd, }

        ))
    }
}

pub fn parse_dahlia(input: &str) -> Result<Program> {
    let inputs = DahliaParser::parse(Rule::prog, input)?;
    let input = inputs.single()?;
    DahliaParser::prog(input)
}
