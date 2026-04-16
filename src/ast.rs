use std::collections::HashMap;

use bumpalo::Bump;

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
pub enum Expr<'a> {
    Cast {
        expr: &'a Expr<'a>,
        ty: TypeAtom,
    },

    ArrayLiteral(Vec<&'a Expr<'a>>),
    RecordLiteral(HashMap<Id, &'a Expr<'a>>),

    RationalLiteral(String),
    IntLiteral {
        value: i64,
        base: u8,
    },
    BoolLiteral(bool),

    ArrayAccess {
        array: Id,
        indices: Vec<&'a Expr<'a>>,
    },
    RecordAccess {
        record: &'a Expr<'a>,
        field: Id,
    },

    Application {
        func: Id,
        args: Vec<&'a Expr<'a>>,
    },

    Id(Id),

    BinOp {
        left: &'a Expr<'a>,
        op: InfixOp,
        right: &'a Expr<'a>,
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
pub enum Command<'a> {
    Empty,
    Par(Vec<&'a Command<'a>>),
    Seq(Vec<&'a Command<'a>>),
    Let {
        id: Id,
        ty: Option<Type>,
        value: Option<&'a Expr<'a>>,
    },
    Update {
        lhs: &'a Expr<'a>,
        op: AssignOp,
        rhs: &'a Expr<'a>,
    },
    View {
        id: Id,
        arr_id: Id,
        dims: Vec<View<'a>>,
    },
    Split {
        id: Id,
        arr_id: Id,
        dims: Vec<usize>,
    },
    Return(&'a Expr<'a>),
    IfElse {
        cond: &'a Expr<'a>,
        then: &'a Command<'a>,
        else_: &'a Command<'a>,
    },
    While {
        cond: &'a Expr<'a>,
        pipeline: bool,
        body: &'a Command<'a>,
    },
    For {
        range: ForRange,
        pipeline: bool,
        body: &'a Command<'a>,
        combine: &'a Command<'a>,
    },
    Decorate(String),
    Expr(&'a Expr<'a>),
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
pub enum Def<'a> {
    Func { sig: FuncSig, body: &'a Command<'a> },
    Record { name: Id, fields: Vec<Decl> },
}

#[derive(Debug)]
pub enum Suffix<'a> {
    Rotation(&'a Expr<'a>),
    Aligned { factor: usize, e: &'a Expr<'a> },
}

#[derive(Debug)]
pub struct View<'a> {
    pub suffix: Suffix<'a>,
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
pub struct Program<'a> {
    pub includes: Vec<Include>,
    pub defs: Vec<Def<'a>>,
    pub decors: Vec<&'a Command<'a>>,
    pub decls: Vec<Decl>,
    pub cmd: &'a Command<'a>,
}

impl Command<'_> {
    pub fn smart_par<'a>(cmds: Vec<&'a Command<'a>>, bump: &'a Bump) -> &'a Command<'a> {
        let mut flat = Vec::new();
        for cmd in cmds {
            match cmd {
                Command::Par(cs) => flat.extend(cs),
                Command::Empty => (),
                _ => flat.push(cmd),
            }
        }

        if flat.is_empty() {
            bump.alloc(Command::Empty)
        } else if flat.len() == 1 {
            flat.remove(0)
        } else {
            bump.alloc(Command::Par(flat))
        }
    }

    pub fn smart_seq<'a>(cmds: Vec<&'a Command<'a>>, bump: &'a Bump) -> &'a Command<'a> {
        let mut flat = Vec::new();
        for cmd in cmds {
            match cmd {
                Command::Seq(cs) => flat.extend(cs),
                Command::Empty => (),
                _ => flat.push(cmd),
            }
        }

        if flat.is_empty() {
            bump.alloc(Command::Empty)
        } else if flat.len() == 1 {
            flat.remove(0)
        } else {
            bump.alloc(Command::Seq(flat))
        }
    }
}
