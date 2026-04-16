use bumpalo::Bump;
use dahlia_rs::parser::parse_dahlia;

fn parse_ok(input: &str) {
    let bump = Bump::new();
    assert!(
        parse_dahlia(input, &bump).is_ok(),
        "Failed to parse:\n{}",
        input
    );
}

#[test]
fn numbers() {
    parse_ok("1;");
    parse_ok("1.25;");
    parse_ok("0.25;");
    parse_ok("0x19;");
    parse_ok("014;");
    parse_ok("0x9e3779b9;");
}

#[test]
fn atoms() {
    parse_ok("true;");
    parse_ok("false;");
    parse_ok("true;");
}

#[test]
fn comments() {
    parse_ok(
        r#"
      /* this is a comment
       * on
       * muliple lines
       */
      // this is comment
      x;
      "#,
    );
}

#[test]
fn binops() {
    parse_ok("1 + 2;");
    parse_ok("1 + 2;");
    parse_ok("1 + 2.5;");
    parse_ok("1 + 2 * 3;");
    parse_ok("true == false;");
    parse_ok("1 << 2;");
    parse_ok("1 >> 2;");
    parse_ok("1 % 2;");
    parse_ok("true || false;");
    parse_ok("true && false;");
}

#[test]
fn binop_precedence_order() {
    parse_ok("(1 + 2) * 3;");
    parse_ok("1 + 2 * 3 >= 10 - 5 / 7;");
    parse_ok("1 >> 2 | 3 ^ 4 & 5;");
    parse_ok("1 >= 2 || 4 < 5;");
}

#[test]
fn if_stmt() {
    parse_ok("if (true) {}");
    parse_ok("if (false) { 1 + 2; }");
    parse_ok("if (false) { 1 + 2; }");
}

#[test]
fn decl() {
    parse_ok("decl x: bit<64>;");
    parse_ok("decl x: bool;");
    parse_ok("decl x: bit<64>[10 bank 5];");
}

#[test]
fn let_stmt() {
    parse_ok("let x = 1; x + 2;");
    parse_ok("let force = 1; x + 2;");
    parse_ok("let x: bit<32>; x + 2;");
}

#[test]
fn for_loop() {
    parse_ok(
        r#"
      for (let i = 0..10) unroll 5 {
        x + 1;
      }
    "#,
    );
}

#[test]
fn while_loop() {
    parse_ok(
        r#"
      while (false) {
        let x = 1;
        for (let i = 0..10) unroll 5 {
          let y = a[i];
          x + y;
        }
      }
    "#,
    );
}

#[test]
fn combiner_syntax() {
    parse_ok(
        r#"
      for (let i = 0..10) {
      } combine {
      }
    "#,
    );
    parse_ok(
        r#"
      for (let i = 0..10) {
      } combine {
        sum += 10;
        let x = 1;
      }
    "#,
    );
}

#[test]
fn refresh_banks() {
    parse_ok(
        r#"
      x + 1;
      ---
      x + 2;
    "#,
    );
}

#[test]
fn commands() {
    parse_ok("{ x+1; }");
}

#[test]
fn functions() {
    parse_ok(
        r#"
      def foo(a: bit<32>) = {}
      "#,
    );
    parse_ok(
        r#"
      def foo(a: bit<32>): bit<32> = {}
      "#,
    );
    parse_ok(
        r#"
      def foo(a: bit<32>[10 bank 5], b: bool) = {
        bar(1, 2, 3);
      }
      "#,
    );
}

#[test]
fn record_definitions() {
    parse_ok(
        r#"
      record Point {
        x: bit<32>;
        y: bit<32>
      }
      "#,
    );
    parse_ok(
        r#"
      record Point {
        x: int;
        y: bit<32>
      }
      "#,
    );
}

#[test]
fn record_literals() {
    parse_ok(
        r#"
      let res: point = { x = 10; y = 10 };
      "#,
    );
}

#[test]
fn array_literals() {
    parse_ok(
        r#"
      let res: bit<32>[10] = { 1, 2, 3 };
      "#,
    );
}

#[test]
fn records_access() {
    parse_ok(
        r#"
      let k = p.x;
      "#,
    );
    parse_ok(
        r#"
      let k = foo[i].x;
      "#,
    );
    parse_ok(
        r#"
      let k = rec.po.x;
      "#,
    );
}

#[test]
fn imports() {
    parse_ok(
        r#"
      import vivado("print.h") {}
      "#,
    );
    parse_ok(
        r#"
      import c++("print.h") {
        def foo(a: bit<32>);
      }
      "#,
    );
}

#[test]
fn simple_views() {
    parse_ok(r#"view v = a[_ :];"#);
    parse_ok(r#"view v = a[_ : bank 2];"#);
    parse_ok(r#"view v = a[4 * i :];"#);
    parse_ok(r#"view v = a[i + 1! :];"#);
    parse_ok(r#"view v = a[4 * i : +3];"#);
    parse_ok(r#"view v = a[i + 1! : +3];"#);
    parse_ok(r#"view v = a[4 * i : bank 5];"#);
    parse_ok(r#"view v = a[i + 1! : bank 5];"#);
    parse_ok(r#"view v = a[4*i:+3 bank 5];"#);
    parse_ok(r#"view v = a[i + 1! : +3 bank 5];"#);
}

#[test]
fn split_views() {
    parse_ok(r#"split b = a[by 10];"#);
    parse_ok(r#"split b = a[by 10][by 20];"#);
}

#[test]
fn casting() {
    parse_ok(r#"let x = (y as bit<32>);"#);
    parse_ok(r#"let x = (y as float);"#);
    parse_ok(
        r#"
      let x = (0x9e3779b9 as ubit<32>);
      let y = (023615674671 as ubit<32>);
      "#,
    );
}
