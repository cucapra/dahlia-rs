use std::fmt::{Display, Formatter};

use crate::{
    ast::{AssignOp, Ast, Command, CommandId, Def, ExprId, FuncSig, Program, TypeContext, TypeId},
    subtyping::is_subtype,
    type_env::TypeEnv,
};

#[derive(Debug)]
pub enum TypecheckError {
    UnexpectedType,
    NoJoin,
    BinopError,
    NotInBinder,
    ArgLengthMismatch,
    IncorrectAccessDims,
    InvalidShrinkWidth,
    InvalidAlignFactor,
    PipelineError,
    MissingField,
    ExtraFields,
    InvalidSplitFactor,
    AlreadyBound,
}

impl Display for TypecheckError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TypecheckError::UnexpectedType => write!(f, "Unexpected type"),
            TypecheckError::NoJoin => write!(f, "No common supertype found"),
            TypecheckError::BinopError => write!(f, "Invalid binary operation"),
            TypecheckError::NotInBinder => write!(f, "Expression should be in let binder"),
            TypecheckError::ArgLengthMismatch => write!(f, "Argument length mismatch"),
            TypecheckError::IncorrectAccessDims => {
                write!(f, "Incorrect number of dimensions for array access")
            }
            TypecheckError::InvalidShrinkWidth => write!(f, "Invalid shrink width"),
            TypecheckError::InvalidAlignFactor => write!(f, "Invalid align factor"),
            TypecheckError::PipelineError => write!(f, "Pipeline error"),
            TypecheckError::MissingField => write!(f, "Missing field in struct literal"),
            TypecheckError::ExtraFields => write!(f, "Extra fields in struct literal"),
            TypecheckError::InvalidSplitFactor => write!(f, "Invalid split factor"),
            TypecheckError::AlreadyBound => write!(f, "Type is already bound"),
        }
    }
}

impl std::error::Error for TypecheckError {}

pub fn typecheck(
    program: &Program,
    ast: &mut Ast,
    tcx: &mut TypeContext,
) -> Result<(), TypecheckError> {
    let mut env = TypeEnv::new();

    let all_defs: Vec<_> = program
        .includes
        .iter()
        .flat_map(|include| &include.defs)
        .chain(&program.defs)
        .collect();

    for def in all_defs {
        check_def(def, &mut env, ast, tcx)?;
    }

    // check the main command
    check_def(
        &Def::Func {
            sig: FuncSig {
                name: ast.ids.push("main".to_string()),
                args: vec![],
                ret_ty: tcx.get_void(),
            },
            body: program.cmd,
        },
        &mut env,
        ast,
        tcx,
    )?;

    Ok(())
}

fn check_def(
    def: &Def,
    env: &mut TypeEnv,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<(), TypecheckError> {
    match def {
        Def::Record { name, fields } => {
            let resolved_fields = fields
                .iter()
                .map(|(id, ty)| {
                    (
                        id.clone(),
                        env.resolve_type(*ty, tcx).expect("Type resolution failed"),
                    )
                })
                .collect();
            env.add_type(
                name.clone(),
                tcx.get_rec_type(name.clone(), resolved_fields),
            )
            .map_err(|_| TypecheckError::AlreadyBound)?;
        }
        Def::Func { sig, body } => {
            env.with_scope(|env| {
                // add args to env
                for decl in &sig.args {
                    let resolved_ty = env
                        .resolve_type(decl.ty, tcx)
                        .expect("Type resolution failed");
                    tcx.id_type_map.insert(decl.id, resolved_ty);
                    env.add(decl.id, resolved_ty, tcx)
                        .map_err(|_| TypecheckError::AlreadyBound)?;
                }

                // add return type to env
                let resolved_ret_ty = env
                    .resolve_type(sig.ret_ty, tcx)
                    .expect("Type resolution failed");
                env.set_ret_type(resolved_ret_ty);

                check_command(*body, env, ast, tcx)?;
                Ok(())
            })?;

            // add function type to env
            // should add to id map as well?
            env.add(
                sig.name,
                tcx.get_func(sig.args.iter().map(|decl| decl.ty).collect(), sig.ret_ty),
                tcx,
            )
            .map_err(|_| TypecheckError::AlreadyBound)?;
        }
    }
    Ok(())
}

fn check_command(
    cmd: CommandId,
    env: &mut TypeEnv,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<(), TypecheckError> {
    match &ast.commands[cmd] {
        Command::Empty => Ok(()),
        Command::Par(cmds) | Command::Seq(cmds) => {
            for cmd in cmds {
                check_command(*cmd, env, ast, tcx)?;
            }
            Ok(())
        }
        Command::IfElse { cond, then, else_ } => {
            let cond_ty = check_expr(*cond, env, ast, tcx)?;
            if cond_ty != tcx.get_bool() {
                return Err(TypecheckError::UnexpectedType);
            }

            env.with_scope(|env| check_command(*then, env, ast, tcx))?;
            env.with_scope(|env| check_command(*else_, env, ast, tcx))?;

            Ok(())
        }
        Command::While {
            cond,
            pipeline: _pipeline,
            body,
        } => {
            // TODO: check pipeline
            let cond_ty = check_expr(*cond, env, ast, tcx)?;
            if cond_ty != tcx.get_bool() {
                return Err(TypecheckError::UnexpectedType);
            }

            env.with_scope(|env| check_command(*body, env, ast, tcx))?;

            Ok(())
        }
        Command::Update { lhs, op: _op, rhs } => {
            let lhs_ty = check_expr(*lhs, env, ast, tcx)?;
            let rhs_ty = check_expr(*rhs, env, ast, tcx)?;

            if !is_subtype(rhs_ty, lhs_ty, tcx) {
                return Err(TypecheckError::UnexpectedType);
            }
            Ok(())
        }
        _ => unimplemented!(),
    }
}

fn check_expr(
    expr: ExprId,
    env: &mut TypeEnv,
    ast: &Ast,
    tcx: &mut TypeContext,
) -> Result<TypeId, TypecheckError> {
    let expr = ast.exprs.get(expr).expect("Expression not found in ast");

    match expr {
        _ => unimplemented!(),
    }
}
