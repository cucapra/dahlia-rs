use std::cell::RefCell;

use anyhow::Error;
use dahlia_rs::{
    ast::Context,
    parser::parse_dahlia,
    typecheck::{TypecheckError, typecheck},
};

fn parse_and_typecheck(input: &str) -> Result<(), Error> {
    let context = RefCell::new(Context::new());
    let program = parse_dahlia(input, &context).expect("program should parse");
    let mut context = context.into_inner();
    typecheck(&program, &mut context)
}

fn typecheck_ok(input: &str) {
    assert!(parse_and_typecheck(input).is_ok());
}

fn same_error_kind(actual: &TypecheckError, expected: &TypecheckError) -> bool {
    match (actual, expected) {
        (TypecheckError::Unsupported(_), TypecheckError::Unsupported(_)) => true,
        _ => actual == expected,
    }
}

fn typecheck_err(input: &str, expected: TypecheckError) {
    let err = parse_and_typecheck(input).expect_err("program should fail to typecheck");
    let actual = err
        .downcast_ref::<TypecheckError>()
        .expect("error should be a TypecheckError");

    assert!(same_error_kind(actual, &expected));
}

fn parse_err(input: &str) {
    let context = RefCell::new(Context::new());
    assert!(parse_dahlia(input, &context).is_err());
}

#[test]
fn let_bindings() {
    typecheck_err(
        "decl a: bit<16>; let x: bit<8> = a;",
        TypecheckError::UnexpectedType,
    );
    typecheck_err(
        "decl a: fix<16,8>; let x: fix<8,4> = a;",
        TypecheckError::UnexpectedType,
    );
    typecheck_err("let x: bit<16> = true;", TypecheckError::UnexpectedType);
    typecheck_err("let x: fix<2,2> = 2+2.1;", TypecheckError::NoJoin);
    typecheck_err("let x: bit<16>; x := true;", TypecheckError::UnexpectedType);
    typecheck_err(
        "let x: fix<2,2>; x := true;",
        TypecheckError::UnexpectedType,
    );
    typecheck_err("let x: fix<2,1>; x := 2;", TypecheckError::UnexpectedType);

    typecheck_ok("let x: ubit<32> = 0x9e3779b9;");
    typecheck_ok("let x: bit<64> = 0x7fffffffffffffff;");
    typecheck_ok("let x: bit<16>;");
    typecheck_ok("let x: bit<16>; x := 1;");
    typecheck_ok("let x: fix<2,1>; x := 1.1;");
}

#[test]
fn unbound_array_access_and_scoping() {
    typecheck_err("x + 1;", TypecheckError::Unbound);

    typecheck_err(
        r#"
          decl a: bit<10>[10];
          a[true];
        "#,
        TypecheckError::UnexpectedType,
    );
    typecheck_err(
        r#"
          decl a: bit<10>[10];
          a[1][1];
        "#,
        TypecheckError::IncorrectAccessDims,
    );
    typecheck_err(
        r#"
          decl a: bit<10>[10][10];
          a[1];
        "#,
        TypecheckError::IncorrectAccessDims,
    );

    typecheck_err("let x = 1; let x = 1;", TypecheckError::AlreadyBound);
    typecheck_err(
        "let x = 1; if(true) { let x = 1; }",
        TypecheckError::AlreadyBound,
    );
    typecheck_err("if (true) {let x = 1;} x + 2;", TypecheckError::Unbound);
    typecheck_err(
        "for (let i = 0..10){let x = 1;} x + 2;",
        TypecheckError::Unbound,
    );
    typecheck_err("while (true) {let x = 1;} x + 2;", TypecheckError::Unbound);

    typecheck_ok(
        r#"
      for (let i = 0..1) {
        let x = 10;
      }
      for (let i = 0..1) {
        let x = 10;
      }
    "#,
    );

    typecheck_err(
        r#"
    {
      let x = 10;
    }
    x;
    "#,
        TypecheckError::Unbound,
    );
}

#[test]
fn binary_operations_and_reassign() {
    typecheck_ok("if (2.5 < 23.5) { 1; }");
    typecheck_ok("let x: float = 1.0; x < 10.0;");
    typecheck_ok("decl x: bit<64>; let y = 1; x + y;");
    typecheck_ok("decl x: fix<64,32>; let y = 1.5 + x;");
    typecheck_ok("decl f: float; let y = 1.5; f + y;");
    typecheck_ok("let x = 1; let y = 2; let z = x + y;");
    typecheck_ok("decl x: bit<32>; decl y: bit<16>; let z = x + y;");
    typecheck_ok("decl x: fix<32,16>; decl y: fix<16,8>; let z = x + y;");
    typecheck_ok("let x: ubit<32> = 0x9e3779b9; let y = x + 0;");
    typecheck_ok("decl x: bit<32>; decl y: bit<16>; x := y;");

    typecheck_err("decl x: fix<64,32>; let y = 1 + x;", TypecheckError::NoJoin);
    typecheck_err("1 + 2.5;", TypecheckError::NoJoin);
    typecheck_err("decl f: float; f + 1;", TypecheckError::NoJoin);
    typecheck_err("decl f: fix<32,16>; f + 1;", TypecheckError::NoJoin);
    typecheck_err(
        "decl a: bit<10>[10]; decl b: bit<10>[10]; a == b;",
        TypecheckError::BinopError,
    );
    typecheck_err("10.5 << 1;", TypecheckError::BinopError);
    typecheck_err("1 || 2;", TypecheckError::BinopError);
    typecheck_err("let x = 1; x := 2.5;", TypecheckError::UnexpectedType);
}

#[test]
fn conditionals_loops_combine_and_sequencing() {
    typecheck_err("if (1) { let x = 10; }", TypecheckError::UnexpectedType);

    typecheck_ok(
        r#"
      while (true) {
        let x = 1;
      }
    "#,
    );
    typecheck_ok(
        r#"
      decl a: bit<64>[10 bank 10];
      let sum: bit<64> = 0;
      for (let i = 0..10) unroll 10 {
        let v = a[i];
      } combine {
        sum += v;
      }
    "#,
    );
    typecheck_ok(
        r#"
      decl a: bit<64>[10];
      let sum: bit<64> = 0;
      for (let i = 0..10) {
        let x = a[i];
      } combine {
        sum += x;
      }
    "#,
    );
    typecheck_ok(
        r#"
      decl a: bit<64>[10 bank 5];
      let sum: bit<64> = 0;
      for (let i = 0..10) unroll 5 {
        let x = a[i];
      } combine {
        sum += x;
      }
    "#,
    );
    typecheck_ok(
        r#"
      let bucket_idx = 10;
      ---
      bucket_idx := (20 as bit<4>);
    "#,
    );
    typecheck_ok(
        r#"
      let test_var:bit<32> = 10;
      {
        test_var := 50;
        ---
        test_var := 30;
      }
    "#,
    );
}

#[test]
fn functions_and_applications() {
    typecheck_err(
        r#"
          def foo(a: bool, a: bit<10>) = {}
        "#,
        TypecheckError::AlreadyBound,
    );
    typecheck_err(
        r#"
          def bar(a: bool) = { foo(a); }
          def foo(a: bool) = { foo(a); }
        "#,
        TypecheckError::Unbound,
    );
    typecheck_err(
        r#"
          def bar(a: bool) = { bar(a); }
        "#,
        TypecheckError::Unbound,
    );
    typecheck_err(
        r#"
          def foo(): bool = { return 5; }
        "#,
        TypecheckError::UnexpectedType,
    );
    typecheck_err(
        r#"
          def bar(a: bool) = { }
          bar(1);
        "#,
        TypecheckError::UnexpectedType,
    );
    typecheck_err(
        r#"
          def foo(a: bit<32>[10 bank 5]) = {
          }
          decl b: bit<32>[5 bank 5];
          foo(b);
        "#,
        TypecheckError::UnexpectedType,
    );
    typecheck_err(
        r#"
          def foo(a: bit<32>, b: bit<32>) = {
          }
          foo(1);
        "#,
        TypecheckError::ArgLengthMismatch,
    );

    typecheck_ok(
        r#"
      def foo(): bit<10> = { return 5; }
      let res: bit<10> = foo();
    "#,
    );
    typecheck_ok(
        r#"
      record point { x: bit<32> }
      def f(p: point): point = {
          let np: point = { x=p.x + 1 };
          return np;
      }
    "#,
    );
    typecheck_ok(
        r#"
      def bar(a: bool) = { }
      let tre: bool;
      for (let i = 0..10) unroll 5 {
        bar(tre);
      }
    "#,
    );
}

#[test]
fn views_and_splits() {
    typecheck_err(
        r#"
          decl a: bit<10>[10 bank 5][10 bank 5];
          view v = a[5 * i :];
        "#,
        TypecheckError::IncorrectAccessDims,
    );
    typecheck_err(
        r#"
          decl a: bool[10 bank 5];
          view v = a[0!:];
          v[3] + 1;
        "#,
        TypecheckError::NoJoin,
    );
    typecheck_err(
        r#"
          decl a: bool[10 bank 5][10 bank 5];
          view v = a[0!:][0!:];
          v[1];
        "#,
        TypecheckError::IncorrectAccessDims,
    );

    typecheck_err(
        r#"
          decl a: bit<32>[10 bank 5][2];
          split v = a[by 5];
        "#,
        TypecheckError::IncorrectAccessDims,
    );
    typecheck_err(
        r#"
          decl a: bit<32>[10];
          split v = a[by 5];
        "#,
        TypecheckError::InvalidSplitFactor,
    );
    typecheck_err(
        r#"
          decl a: bit<32>[10 bank 5];
          split v = a[by 5];
          v[0];
        "#,
        TypecheckError::IncorrectAccessDims,
    );

    typecheck_err(
        r#"
          decl x: bit<32>;
          decl a: bit<10>[10 bank 5];
          view v = a[3 * x :];
        "#,
        TypecheckError::InvalidAlignFactor,
    );
    typecheck_err(
        r#"
          decl x: bit<32>;
          decl a: bit<10>[10 bank 10];
          view v = a[5 * x :];
        "#,
        TypecheckError::InvalidAlignFactor,
    );

    parse_err(
        r#"
      decl x: bit<32>;
      decl a: bit<10>[10 bank 5];
      view v = a[x * x :];
    "#,
    );

    typecheck_ok(
        r#"
      decl a: bit<10>[16 bank 8];
      for (let i = 0..4) {
        view v = a[8 * i :];
      }
    "#,
    );
    typecheck_ok(
        r#"
      let A: float[10 bank 2];
      view m1 = A[_: bank 2];
      m1[0];
      ---
      m1[1];
    "#,
    );
    typecheck_ok(
        r#"
      decl a: bit<32>[10 bank 5];
      split v = a[by 5];
    "#,
    );
    typecheck_ok(
        r#"
      decl x: bit<32>;
      decl a: bit<10>[16 bank 8];
      view v = a[6 * x : bank 2];
    "#,
    );
    typecheck_ok(
        r#"
      decl a: bit<10>[10 bank 5];
      decl i: bit<32>;
      view v = a[i * i ! :];
    "#,
    );
}

#[test]
fn loop_iterators_and_pipeline() {
    typecheck_ok(
        r#"
      for (let i = 0..10) {
        let x = i * 2;
      }
    "#,
    );
    typecheck_ok(
        r#"
      let temp = 0;
      for (let i = 0..10) {
        if (i == temp) {
          let x = 0;
        }
      }
    "#,
    );
    typecheck_ok(
        r#"
      def test(a: bit<32>) = {
        let test2 = a;
      }

      for (let i = 0..5) {
        test(i);
      }
    "#,
    );
    typecheck_ok(
        r#"
      for (let i = 0..10) {
        let x = i | 2;
      }
    "#,
    );
    typecheck_ok(
        r#"
      decl a: bit<10>[10];
      for (let i = 0..10) {
        a[i * 2];
      }
    "#,
    );
    typecheck_ok(
        r#"
      for (let i = 0..4) pipeline {
        let a = 1 + 2;
        let b = 3 + 4;
      }
    "#,
    );
    typecheck_ok(
        r#"
      let x = 10;
      while (x < 100) pipeline {
        let a = 1 + 2;
        let b = 3 + 4;
      }
    "#,
    );

    typecheck_err(
        r#"
          for (let i = 0..4) pipeline {
            let a = 1 + 2;
            ---
            let b = 3 + 4;
          }
        "#,
        TypecheckError::PipelineError,
    );
    typecheck_err(
        r#"
          let x = 10;
          while (x < 100) pipeline {
            let a = 1 + 2;
            ---
            let b = 3 + 4;
          }
        "#,
        TypecheckError::PipelineError,
    );
}

#[test]
fn records() {
    typecheck_err(
        r#"
          record bars {
            k: point
          }
        "#,
        TypecheckError::UnknownAlias,
    );
    typecheck_err(
        r#"
          record bars {
            k: bit<32>
          }
          record bars {
            l: bit<32>
          }
        "#,
        TypecheckError::AlreadyBound,
    );
    typecheck_err(
        r#"
          record point { x: bit<32>; y: bit<32> }
          let p = {x = 1; y = 2 };
        "#,
        TypecheckError::ExplicitTypeMissing,
    );
    typecheck_err(
        r#"
          record point { x: bit<32>; y: bit<32> }
          let p = 1 + {x = 1; y = 2 };
        "#,
        TypecheckError::NotInBinder,
    );
    typecheck_err(
        r#"
          record point { x: bit<32>; y: bit<32> }
          let p: point = {x = 1};
        "#,
        TypecheckError::MissingField,
    );
    typecheck_err(
        r#"
          record point { x: bit<32> }
          let p: point = {x = 1; y = 2};
        "#,
        TypecheckError::ExtraFields,
    );

    typecheck_ok(
        r#"
      record point {
        x: bit<32>;
        y: bit<32>
      }
    "#,
    );
    typecheck_ok(
        r#"
      record point {
        x: bit<32>;
        y: bit<32>
      }
      decl k: point;
    "#,
    );
    typecheck_ok(
        r#"
      record point {
        x: bit<32>;
        y: bit<32>
      }
      record bars {
        k: point
      }
    "#,
    );
    typecheck_ok(
        r#"
      record point {
        x: bit<32>
      }
      decl k: point;
      let x = k.x;
    "#,
    );
    typecheck_ok(
        r#"
      record point {
        x: bit<32>
      }
      decl k: point;
      let x = k.x + 1;
    "#,
    );
    typecheck_ok(
        r#"
      record point {
        x: bit<32>
      }
      record foo {
        p: point
      }
      decl k: foo;
      let x = k.p.x + 1;
    "#,
    );
    typecheck_ok(
        r#"
      record point { x: ubit<32> }
      let a: point = {x=10};
      let b: point = (a as point);
    "#,
    );
    typecheck_ok(
        r#"
      record point { x: bit<32>; y: bit<32> }
      let p: point = {x = 1; y = 2 };
    "#,
    );
    typecheck_ok(
        r#"
      record point { x: bit<32>; y: bit<32> }
      let p: point = {x = 1; y = 2 };
      let f: bit<32> = p.x;
    "#,
    );
}

#[test]
fn array_literals() {
    typecheck_err(
        r#"
          let x = {1, 2, 3};
        "#,
        TypecheckError::ExplicitTypeMissing,
    );
    typecheck_err(
        r#"
          let x: bit<32>[10][10] = {1, 2, 3};
        "#,
        TypecheckError::Unsupported(""),
    );
    typecheck_err(
        r#"
          let x: bit<32>[5] = {1, 2, 3};
        "#,
        TypecheckError::LiteralLengthMismatch,
    );
    typecheck_err(
        r#"
          let x: bit<32>[3] = {true, false, true};
        "#,
        TypecheckError::UnexpectedType,
    );

    typecheck_ok(
        r#"
      let x: bool[3 bank 3] = {true, false, true};
    "#,
    );
    typecheck_ok(
        r#"
      let x: bool[3] = {true, false, true};
      {
        x[1];
        ---
        x[0] := false;
      }
    "#,
    );
}

#[test]
fn indexing_subtyping_imports_and_casting() {
    typecheck_ok("decl a: bit<10>[10]; decl x: bit<10>; a[x] := 5;");

    typecheck_ok("1 == 2;");
    typecheck_ok(
        r#"
      decl x: bit<16>;
      decl y: bit<32>;
      x == y;
    "#,
    );
    typecheck_ok(
        r#"
      for (let i = 0..12) {
        i == 1;
      }
    "#,
    );
    typecheck_ok(
        r#"
      decl x: bit<32>;
      for (let i = 0..12) {
        i == x;
      }
    "#,
    );
    typecheck_ok(
        r#"
      decl arr:bit<32>[10];
      for (let i = 0..33) {
        arr[5] := i * 1;
      }
    "#,
    );
    typecheck_ok(
        r#"
      for (let i = 0..10) {
        for (let j = 0..2) {
            let x = i + j;
          }
      }
    "#,
    );
    typecheck_ok(
        r#"
      let x = true;
      let y = false;
      x == y;

      let i1: bit<32> = 10;
      let i2: bit<32> = 11;
      i1 == i2;
    "#,
    );
    typecheck_ok(
        r#"
      decl x: float;
      decl y: float;
      let z: double = x + y;
    "#,
    );
    typecheck_ok(
        r#"
      import vivado("print.cpp") {
        def print_vect(f: float[4]);
      }
      decl a: float[4];
      print_vect(a);
    "#,
    );

    typecheck_err(
        r#"
          def foo(b: bit<10>[10]) = {
            b[0] := 10;
          }

          decl a: bit<2>[10];
          foo(a);
        "#,
        TypecheckError::UnexpectedType,
    );
    typecheck_err(
        r#"
          decl x: fix<3,2>;
          decl y: fix<3,2>;
          let z: double = x + y;
        "#,
        TypecheckError::UnexpectedType,
    );
    typecheck_err(
        r#"
          decl x: ubit<32>;
          decl y: bit<32>;
          let z = x + y;
        "#,
        TypecheckError::NoJoin,
    );
    typecheck_err(
        r#"
          decl x: ufix<32,16>;
          decl y: fix<32,16>;
          let z = x + y;
        "#,
        TypecheckError::NoJoin,
    );
    typecheck_err(
        r#"
          let z:ufix<32,16> = -0.5;
        "#,
        TypecheckError::UnexpectedType,
    );

    typecheck_ok(
        r#"
      decl x: float;
      decl y: bit<32>;
      (y as float) + x;
    "#,
    );
    typecheck_ok(
        r#"
      decl x: fix<32,16>;
      decl y: bit<16>;
      (y as fix<32,16>) + x;
    "#,
    );
    typecheck_ok(
        r#"
      decl x: float;
      decl y: bit<32>;
      (x as bit<32>) + y;
    "#,
    );
    typecheck_ok(
        r#"
      decl x: fix<32,16>;
      decl y: bit<16>;
      (x as bit<16>) + y;
    "#,
    );
    typecheck_ok(
        r#"
      decl x: float;
      decl y: double;
      y + (x as double);
    "#,
    );
    typecheck_ok(
        r#"
      decl x: fix<10,5>;
      decl y: double;
      y + (x as double);
    "#,
    );
}
