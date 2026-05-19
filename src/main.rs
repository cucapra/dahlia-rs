use std::{
    cell::RefCell,
    env,
    io::{self, Read},
};

use anyhow::{Result, anyhow};
use calyx_frontend::Workspace;
use calyx_ir::{self as ir, Printer};
use dahlia_rs::{
    ast::Context, calyx_backend::emit_calyx, parser::parse_dahlia, pretty::pretty_program,
    resolver::resolve_names, typecheck::typecheck,
};

fn main() -> Result<()> {
    let mut source = String::new();
    io::stdin().read_to_string(&mut source)?;

    let dahlia_ctx = RefCell::new(Context::new());

    let program = parse_dahlia(&source, &dahlia_ctx)?;

    let mut dahlia_ctx = dahlia_ctx.into_inner();
    resolve_names(&program, &mut dahlia_ctx)?;
    
    typecheck(&program, &mut dahlia_ctx)?;

    // ast_debug(&context, &program);

    let _pretty = pretty_program(&program, &dahlia_ctx);
    // println!("{}", _pretty);

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
