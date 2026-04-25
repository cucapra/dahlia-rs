use std::cell::RefCell;
use std::collections::HashMap;

use cranelift_entity::{EntityList, ListPool, PrimaryMap, entity_impl};

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct ExprId(u32);
entity_impl!(ExprId, "expr");

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct CommandId(u32);
entity_impl!(CommandId, "cmd");

#[derive(Debug, Default)]
pub struct AstData {
    pub exprs: PrimaryMap<ExprId, Expr>,
    pub commands: PrimaryMap<CommandId, Command>,
    pub expr_lists: ListPool<ExprId>,
    pub command_lists: ListPool<CommandId>,
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct Id(pub String);

#[derive(Debug)]
pub enum TypeAtom {
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
    Alias(Id),
}

#[derive(Debug)]
pub struct DimSpec {
    pub length: usize,
    pub bank: Option<usize>,
}

#[derive(Debug)]
pub enum Type {
    Simple(TypeAtom),
    Array {
        element_type: TypeAtom,
        dims: Vec<DimSpec>,
        ports: usize,
    },
}

#[derive(Debug)]
pub enum InfixOp {
    Mul,
    Div,
    Mod,
    Add,
    Sub,
    Shl,
    Shr,
    Eq,
    Neq,
    Le,
    Ge,
    Lt,
    Gt,
    And,
    Or,
    Band,
    Bor,
    Bxor,
}

#[derive(Debug)]
pub enum Expr {
    Cast {
        expr: ExprId,
        ty: TypeAtom,
    },

    ArrayLiteral(EntityList<ExprId>),
    RecordLiteral(HashMap<Id, ExprId>),

    RationalLiteral(String),
    IntLiteral {
        value: i64,
        base: u8,
    },
    BoolLiteral(bool),

    ArrayAccess {
        array: Id,
        indices: EntityList<ExprId>,
    },
    RecordAccess {
        record: ExprId,
        field: Id,
    },

    Application {
        func: Id,
        args: EntityList<ExprId>,
    },

    Id(Id),

    BinOp {
        left: ExprId,
        op: InfixOp,
        right: ExprId,
    },
}

#[derive(Debug)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
}

#[derive(Debug)]
pub struct ForRange {
    pub id: Id,
    pub ty: Option<Type>,
    pub rev: bool,
    pub start: usize,
    pub end: usize,
    pub unroll: usize,
}

#[derive(Debug)]
pub enum Command {
    Empty,
    Par(EntityList<CommandId>),
    Seq(EntityList<CommandId>),
    Let {
        id: Id,
        ty: Option<Type>,
        value: Option<ExprId>,
    },
    Update {
        lhs: ExprId,
        op: AssignOp,
        rhs: ExprId,
    },
    View {
        id: Id,
        arr_id: Id,
        dims: Vec<View>,
    },
    Split {
        id: Id,
        arr_id: Id,
        dims: Vec<usize>,
    },
    Return(ExprId),
    IfElse {
        cond: ExprId,
        then: CommandId,
        else_: CommandId,
    },
    While {
        cond: ExprId,
        pipeline: bool,
        body: CommandId,
    },
    For {
        range: ForRange,
        pipeline: bool,
        body: CommandId,
        combine: CommandId,
    },
    Decorate(String),
    Expr(ExprId),
}

#[derive(Debug)]
pub struct Decl {
    pub id: Id,
    pub ty: Type,
}

#[derive(Debug)]
pub struct FuncSig {
    pub name: Id,
    pub args: Vec<Decl>,
    pub ret_ty: Option<Type>,
}

#[derive(Debug)]
pub enum Def {
    Func { sig: FuncSig, body: CommandId },
    Record { name: Id, fields: Vec<Decl> },
}

#[derive(Debug)]
pub enum Suffix {
    Rotation(ExprId),
    Aligned { factor: usize, e: ExprId },
}

#[derive(Debug)]
pub struct View {
    pub suffix: Suffix,
    pub prefix: Option<usize>,
    pub shrink: Option<usize>,
}

#[derive(Debug)]
pub enum Backend {
    Cpp,
    Vivado,
    Futil,
    Calyx,
}

#[derive(Debug)]
pub struct Include {
    pub backends: Vec<(Backend, String)>,
    pub defs: Vec<FuncSig>,
}

#[derive(Debug)]
pub struct Program {
    pub includes: Vec<Include>,
    pub defs: Vec<Def>,
    pub decors: EntityList<CommandId>,
    pub decls: Vec<Decl>,
    pub cmd: CommandId,
}

impl Command {
    pub fn smart_par(cmds: Vec<CommandId>, ast_data: &RefCell<AstData>) -> CommandId {
        let mut flat = Vec::new();
        {
            let ast_data = ast_data.borrow();
            for cmd in cmds {
                match &ast_data.commands[cmd] {
                    Command::Par(cs) => flat.extend(cs.as_slice(&ast_data.command_lists)),
                    Command::Empty => (),
                    _ => flat.push(cmd),
                }
            }
        }

        if flat.is_empty() {
            ast_data.borrow_mut().commands.push(Command::Empty)
        } else if flat.len() == 1 {
            flat.remove(0)
        } else {
            let mut ast_data = ast_data.borrow_mut();
            let cmds = EntityList::from_iter(flat, &mut ast_data.command_lists);
            ast_data.commands.push(Command::Par(cmds))
        }
    }

    pub fn smart_seq(cmds: Vec<CommandId>, ast_data: &RefCell<AstData>) -> CommandId {
        let mut flat = Vec::new();
        {
            let ast_data = ast_data.borrow();
            for cmd in cmds {
                match &ast_data.commands[cmd] {
                    Command::Seq(cs) => flat.extend(cs.as_slice(&ast_data.command_lists)),
                    Command::Empty => (),
                    _ => flat.push(cmd),
                }
            }
        }

        if flat.is_empty() {
            ast_data.borrow_mut().commands.push(Command::Empty)
        } else if flat.len() == 1 {
            flat.remove(0)
        } else {
            let mut ast_data = ast_data.borrow_mut();
            let cmds = EntityList::from_iter(flat, &mut ast_data.command_lists);
            ast_data.commands.push(Command::Seq(cmds))
        }
    }
}
