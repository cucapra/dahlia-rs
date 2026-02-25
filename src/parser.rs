use pest::pratt_parser::PrattParser;
use pest_consume::{Error, Parser, match_nodes};

use crate::ast::{
    AssignOp, Command, Decl, Def, DimSpec, Expr, Id, InfixOp, Program, Type, TypeAtom,
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

    fn assign_op(input: Node) -> Result<AssignOp> {
        Ok(match input.as_rule() {
            Rule::assign => AssignOp::Assign,
            Rule::add_assign => AssignOp::AddAssign,
            Rule::sub_assign => AssignOp::SubAssign,
            Rule::mul_assign => AssignOp::MulAssign,
            Rule::div_assign => AssignOp::DivAssign,
            _ => unreachable!("Unexpected assignment operator: {:?}", input.as_rule()),
        })
    }

    fn update(input: Node) -> Result<Command> {
        Ok(match_nodes!(input.into_children();
            [expr(lhs), assign_op(op), expr(rhs)] => Command::Update{lhs, op, rhs}
        ))
    }

    fn simple_cmd(input: Node) -> Result<Command> {
        match_nodes!(input.into_children();
            [let_stmt(cmd)] => Ok(cmd),
            [update(cmd)] => Ok(cmd),
            [expr(e)] => Ok(Command::Expr(e))
        )
    }

    fn cmd(input: Node) -> Result<Command> {
        match_nodes!(input.into_children();
            [par_cmd(c)..] => Ok(Command::Par(c.into_iter().collect())),
        )
    }

    fn par_cmd(input: Node) -> Result<Command> {
        match_nodes!(input.into_children();
            [cmd(c)] => Ok(c),
            [simple_cmd(c)..] => Ok(Command::Par(c.into_iter().collect()))
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
            [iden(name), func_args(args), ty(ret_ty), cmd(body)] => Def::Func{name, args, ret_ty: Some(ret_ty), body},
            [iden(name), func_args(args), cmd(body)] => Def::Func{name, args, ret_ty: None, body}
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

    fn prog_cmd(input: Node) -> Result<Option<Command>> {
        Ok(match_nodes!(input.into_children();
            [cmd(cmd)] => Some(cmd),
            [] => None
        ))
    }

    fn prog(input: Node) -> Result<Program> {
        Ok(match_nodes!(input.into_children();
            [defs(defs), decls(decls), prog_cmd(cmd), EOI(_)] => Program{defs, decls, cmd}
        ))
    }
}

pub fn parse_dahlia(input: &str) -> Result<Program> {
    let inputs = DahliaParser::parse(Rule::prog, input)?;
    let input = inputs.single()?;
    DahliaParser::prog(input)
}
