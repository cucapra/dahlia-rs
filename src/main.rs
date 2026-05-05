use std::cell::RefCell;

use dahlia_rs::{ast::Context, ast_debug::ast_debug, parser::parse_dahlia, typecheck::typecheck};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let source = r###"
    record point {
  x: bit<32>;
  y: bit<32>
}

decl shape1: point[2];
decl shape2: point[2];
decl result: point;

let X: bit<32> = 0;
let Y: bit<32> = 0;

for (let i = 0..2) {
  let x = shape1[i].x + shape2[i].x;
  let y = shape1[i].y + shape2[i].y;
} combine {
  X += x;
  Y += y;
}

let out: point = { x = X; y = Y };
result := out;
    "###;
    let context = RefCell::new(Context::default());
    let program = parse_dahlia(source, &context)?;

    let mut context = context.take();

    typecheck(&program, &mut context.ast, &mut context.tcx)?;
    ast_debug(&context, &program);

    Ok(())
}
