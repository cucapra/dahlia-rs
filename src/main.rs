use std::{cell::RefCell, env, path::Path};

use anyhow::{Result, anyhow};
use calyx_frontend::Workspace;
use calyx_ir::{self as ir, Printer};
use dahlia_rs::{
    ast::{Command, Context, Expr}, calyx_backend::emit_calyx, parser::parse_dahlia, pretty::pretty_program, resolver::resolve_names, typecheck::typecheck
};

fn main() -> Result<()> {
    /* let source = r###"
import futil("primitives/math.futil") { def sqrt(in: ubit<32>): ubit<32>; }

def foo(a: ubit<32>): ubit<32> = {
  let temp: ubit<32> = a;
  return temp;
}

let b: ubit<32> = (1 as ubit<32>);
let c: ubit<32> = foo(b);
let d: ubit<32> = sqrt(c);
    "###;
    */
    //     let source = r###"
    //     record point {
    //   x: bit<32>;
    //   y: bit<32>
    // }

    // decl shape1: point[2];
    // decl shape2: point[2];
    // decl result: point;

    // let X: bit<32> = 0;
    // let Y: bit<32> = 0;

    // for (let i = 0..2) {
    //   let x = shape1[i].x + shape2[i].x;
    //   let y = shape1[i].y + shape2[i].y;
    // } combine {
    //   X += x;
    //   Y += y;
    // }

    // let out: point = { x = X; y = Y };
    // result := out;
    //     "###;

    let source = r###"
// record point { x: bit<32>; y: bit<32> }
// let p: point = {x = 1; y = 2 };
record point { x: bit<32>; y: bit<32> }
          let p: point = {x = 1; y = 2 };
          let f: bit<32> = p.x;
// record point { x: ubit<32> }
// let a: point = {x=10};
// let b: point = (a as point);
    "###;
    let dahlia_ctx = RefCell::new(Context::new());
    let program = parse_dahlia(source, &dahlia_ctx)?;

    let mut dahlia_ctx = dahlia_ctx.into_inner();

    resolve_names(&program, &mut dahlia_ctx)?;
    typecheck(&program, &mut dahlia_ctx)?;
    // ast_debug(&context, &program);
    // println!("{}", pretty_program(&program, &context));

    // should point to the root of the cloned Calyx repo
    let calyx_root = env::var("CALYX_ROOT").expect("CALYX_ROOT environment variable not set");

    let ws = Workspace::construct(
        &Some("resources/stdlib.futil".into()),
        &Path::new(&calyx_root),
    )
    .map_err(|e| anyhow!("failed to construct Calyx workspace {:?}", e))?;
    let mut calyx_ast = ir::from_ast::ast_to_ir(ws)
        .map_err(|e| anyhow!("failed to convert Calyx AST to IR: {:?}", e))?;

    emit_calyx(&program , &mut calyx_ast, &dahlia_ctx)?;

    Printer::write_context(&calyx_ast, false, &mut std::io::stdout())
        .map_err(|e| anyhow!("failed to print Calyx IR: {:?}", e))?;

    Ok(())
}
