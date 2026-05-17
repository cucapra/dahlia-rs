use std::{cell::RefCell, env, path::Path};

use anyhow::{Result, anyhow};
use calyx_frontend::Workspace;
use calyx_ir::{self as ir, Printer};
use dahlia_rs::{
    ast::{Command, Context, Expr},
    calyx_backend::emit_calyx,
    parser::parse_dahlia,
    pretty::pretty_program,
    resolver::resolve_names,
    typecheck::typecheck,
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
// record point { x: bit<32>; y: bit<32> }
//           let p: point = {x = 1; y = 2 };
//           let f: bit<32> = p.x;
// record point { x: ubit<32> }
// let a: point = {x=10};
// let b: point = (a as point);

def foo(a: ubit<32>, b: ubit<32>[4][4]): ubit<32> = {
  let temp: ubit<32> = a;
  return temp;
}

decl a: fix<32,16>[2][2];
decl b: fix<32,16>[2][2];
decl result: fix<32,16>[2][2];
decl foo: bool;
decl bar: bit<8>;
decl baz: fix<16,8>[4][8][16];
{
  let i: bit<1> = (0 as bit<1>);
  ---
  /* @bound(2) */ while ((i <= (1 as bit<1>))) {
    {
      let j: bit<1> = (0 as bit<1>);
      ---
      /* @bound(2) */ while ((j <= (1 as bit<1>))) {
        let a_read0_: fix<32,16> = a[(i as ubit<2>)][(j as ubit<2>)];
        let b_read0_: fix<32,16> = b[(i as ubit<2>)][(j as ubit<2>)];
        ---
        result[(i as ubit<2>)][(j as ubit<2>)] := (a_read0_ + b_read0_);
        ---
        j := (j + (1 as bit<1>));
      }
    }
    ---
    i := (i + (1 as bit<1>));
  }
}
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

    let ws = Workspace::construct(&Some("resources/stdlib.futil".into()), &[calyx_root.into()])
        .map_err(|e| anyhow!("failed to construct Calyx workspace {:?}", e))?;
    let mut calyx_ast = ir::from_ast::ast_to_ir(ws, ir::from_ast::AstConversionConfig::default())
        .map_err(|e| anyhow!("failed to convert Calyx AST to IR: {:?}", e))?;

    // ugly workaround, probably need a better way to populate Calyx primitives
    calyx_ast.components.pop();

    emit_calyx(&program, &mut calyx_ast, &dahlia_ctx)?;

    Printer::write_context(&calyx_ast, true, &mut std::io::stdout())
        .map_err(|e| anyhow!("failed to print Calyx IR: {:?}", e))?;

    Ok(())
}
