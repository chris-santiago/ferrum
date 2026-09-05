//! Vega-style expression evaluator for data transforms.
//!
//! Provides a recursive-descent parser and per-row evaluator for expressions
//! like `"datum.x > 10"` or `"datum.age * 2 + 1"`. Used by `transform_filter`
//! and `transform_calculate` to evaluate predicates and computed columns.

use std::collections::BTreeSet;

// ─── Values ─────────────��───────────────────────────────────────���───────────

/// Runtime value produced by expression evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprValue {
    Number(f64),
    Str(String),
    Bool(bool),
    Null,
    /// Array literal value — produced only by evaluating an array literal
    /// node (`[e1, e2, …]`); never read from a datum column. Exists so
    /// builtin functions like `indexof` can consume array literals as
    /// evaluated expressions rather than a separate non-evaluated AST shape.
    Array(Vec<ExprValue>),
}

impl ExprValue {
    /// Coerce to boolean for conditionals. Numbers: 0/NAN → false. Strings: empty → false.
    pub(crate) fn is_truthy(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::Number(n) => *n != 0.0 && !n.is_nan(),
            Self::Str(s) => !s.is_empty(),
            Self::Null => false,
            Self::Array(items) => !items.is_empty(),
        }
    }

    /// Extract as f64, returning NAN for non-numeric types.
    fn as_number(&self) -> f64 {
        match self {
            Self::Number(n) => *n,
            Self::Bool(true) => 1.0,
            Self::Bool(false) => 0.0,
            _ => f64::NAN,
        }
    }
}

// ─── Error ──────────────────────────────────────────────────────────────────

/// Parse-time error with position information.
#[derive(Debug, Clone, PartialEq)]
pub struct ExprError {
    pub message: String,
    pub position: usize,
}

impl std::fmt::Display for ExprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expr error at position {}: {}", self.position, self.message)
    }
}

impl std::error::Error for ExprError {}

// ─── AST ────────────────────────────────────────────────────────────────────

/// Parsed expression AST. Opaque to callers — use [`Expr::parse`] to create
/// and [`Expr::eval`] to evaluate.
#[derive(Debug, Clone)]
pub struct Expr {
    root: Node,
}

#[derive(Debug, Clone)]
enum Node {
    Literal(ExprValue),
    Array(Vec<Node>),
    Call(String, Vec<Node>),
    DatumAccess(Vec<String>),
    UnaryNeg(Box<Node>),
    UnaryNot(Box<Node>),
    BinaryOp(BinOp, Box<Node>, Box<Node>),
    Ternary(Box<Node>, Box<Node>, Box<Node>),
}

/// Signature every builtin evaluator implements: given the call's
/// already-parsed argument nodes and the current row's datum accessor,
/// produce the call's result.
type BuiltinFn = fn(&[Node], &dyn Fn(&str) -> ExprValue) -> ExprValue;

/// Builtin function table: name -> (arity, eval fn). This is the single
/// point to register a future builtin — add an entry here, pairing the
/// name and arity with its evaluator in one tuple. Because the evaluator
/// lives in the same row as the arity, a new entry cannot compile without
/// its dispatch arm, unlike a separate arity table plus a hand-matched
/// `eval_call` that could silently drift out of sync.
const BUILTINS: &[(&str, usize, BuiltinFn)] = &[("indexof", 2, eval_indexof)];

fn resolve_builtin(name: &str) -> Option<usize> {
    BUILTINS
        .iter()
        .find(|(n, ..)| *n == name)
        .map(|(_, arity, _)| *arity)
}

fn find_builtin(name: &str) -> Option<BuiltinFn> {
    BUILTINS.iter().find(|(n, ..)| *n == name).map(|(_, _, f)| *f)
}

fn supported_builtins() -> String {
    BUILTINS
        .iter()
        .map(|(n, ..)| *n)
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Gt,
    Gte,
    Lt,
    Lte,
    Eq,
    Neq,
    And,
    Or,
}

// ─── Tokenizer ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Str(String),
    Ident(String),
    // Punctuation / operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Gt,
    Gte,
    Lt,
    Lte,
    EqEq,
    BangEq,
    And,
    Or,
    Bang,
    Question,
    Colon,
    Comma,
    Dot,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Eof,
}

#[derive(Debug, Clone)]
struct Located {
    token: Token,
    pos: usize,
}

fn tokenize(input: &str) -> Result<Vec<Located>, ExprError> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        // Skip whitespace
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }

        let start = i;

        match bytes[i] {
            b'+' => {
                tokens.push(Located { token: Token::Plus, pos: start });
                i += 1;
            }
            b'-' => {
                tokens.push(Located { token: Token::Minus, pos: start });
                i += 1;
            }
            b'*' => {
                tokens.push(Located { token: Token::Star, pos: start });
                i += 1;
            }
            b'/' => {
                tokens.push(Located { token: Token::Slash, pos: start });
                i += 1;
            }
            b'%' => {
                tokens.push(Located { token: Token::Percent, pos: start });
                i += 1;
            }
            b'?' => {
                tokens.push(Located { token: Token::Question, pos: start });
                i += 1;
            }
            b':' => {
                tokens.push(Located { token: Token::Colon, pos: start });
                i += 1;
            }
            b',' => {
                tokens.push(Located { token: Token::Comma, pos: start });
                i += 1;
            }
            b'.' => {
                tokens.push(Located { token: Token::Dot, pos: start });
                i += 1;
            }
            b'(' => {
                tokens.push(Located { token: Token::LParen, pos: start });
                i += 1;
            }
            b')' => {
                tokens.push(Located { token: Token::RParen, pos: start });
                i += 1;
            }
            b'[' => {
                tokens.push(Located { token: Token::LBracket, pos: start });
                i += 1;
            }
            b']' => {
                tokens.push(Located { token: Token::RBracket, pos: start });
                i += 1;
            }
            b'>' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    tokens.push(Located { token: Token::Gte, pos: start });
                    i += 2;
                } else {
                    tokens.push(Located { token: Token::Gt, pos: start });
                    i += 1;
                }
            }
            b'<' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    tokens.push(Located { token: Token::Lte, pos: start });
                    i += 2;
                } else {
                    tokens.push(Located { token: Token::Lt, pos: start });
                    i += 1;
                }
            }
            b'=' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    tokens.push(Located { token: Token::EqEq, pos: start });
                    i += 2;
                } else {
                    return Err(ExprError {
                        message: "unexpected '='; did you mean '=='?".into(),
                        position: start,
                    });
                }
            }
            b'!' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    tokens.push(Located { token: Token::BangEq, pos: start });
                    i += 2;
                } else {
                    tokens.push(Located { token: Token::Bang, pos: start });
                    i += 1;
                }
            }
            b'&' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'&' {
                    tokens.push(Located { token: Token::And, pos: start });
                    i += 2;
                } else {
                    return Err(ExprError {
                        message: "unexpected '&'; did you mean '&&'?".into(),
                        position: start,
                    });
                }
            }
            b'|' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                    tokens.push(Located { token: Token::Or, pos: start });
                    i += 2;
                } else {
                    return Err(ExprError {
                        message: "unexpected '|'; did you mean '||'?".into(),
                        position: start,
                    });
                }
            }
            b'"' | b'\'' => {
                let quote = bytes[i];
                i += 1;
                let mut s = String::new();
                while i < bytes.len() && bytes[i] != quote {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 1;
                        match bytes[i] {
                            b'n' => {
                                s.push('\n');
                                i += 1;
                            }
                            b't' => {
                                s.push('\t');
                                i += 1;
                            }
                            b'\\' => {
                                s.push('\\');
                                i += 1;
                            }
                            b'\'' => {
                                s.push('\'');
                                i += 1;
                            }
                            b'"' => {
                                s.push('"');
                                i += 1;
                            }
                            _ => {
                                // Non-ASCII byte after a backslash (e.g. `\é`):
                                // decode the full char rather than treating
                                // its lead byte as Latin-1.
                                let ch = input[i..].chars().next().unwrap();
                                s.push('\\');
                                s.push(ch);
                                i += ch.len_utf8();
                            }
                        }
                    } else {
                        // Decode one UTF-8 char at a time (not one byte at a
                        // time) so multi-byte characters (e.g. "Zürich")
                        // round-trip correctly instead of being mangled into
                        // one mojibake char per byte.
                        let ch = input[i..].chars().next().unwrap();
                        s.push(ch);
                        i += ch.len_utf8();
                    }
                }
                if i >= bytes.len() {
                    return Err(ExprError {
                        message: "unterminated string literal".into(),
                        position: start,
                    });
                }
                i += 1; // closing quote
                tokens.push(Located { token: Token::Str(s), pos: start });
            }
            c if c.is_ascii_digit() || (c == b'.' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit()) => {
                let num_start = i;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                // Handle scientific notation (e.g., 1e10, 2.5E-3)
                if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                    i += 1;
                    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                        i += 1;
                    }
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let num_str = &input[num_start..i];
                let n: f64 = num_str.parse().map_err(|_| ExprError {
                    message: format!("invalid number literal: {num_str}"),
                    position: num_start,
                })?;
                tokens.push(Located { token: Token::Number(n), pos: num_start });
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let id_start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = &input[id_start..i];
                let token = match word {
                    "true" => Token::Ident("true".into()),
                    "false" => Token::Ident("false".into()),
                    "null" => Token::Ident("null".into()),
                    _ => Token::Ident(word.to_string()),
                };
                tokens.push(Located { token, pos: id_start });
            }
            _ => {
                return Err(ExprError {
                    message: format!("unexpected character: '{}'", bytes[i] as char),
                    position: i,
                });
            }
        }
    }

    tokens.push(Located { token: Token::Eof, pos: input.len() });
    Ok(tokens)
}

// ─── Parser ─────────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Located>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Located>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    fn peek_pos(&self) -> usize {
        self.tokens[self.pos].pos
    }

    fn advance(&mut self) -> &Located {
        let t = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, expected: &Token) -> Result<(), ExprError> {
        if self.peek() == expected {
            self.advance();
            Ok(())
        } else {
            Err(ExprError {
                message: format!("expected {:?}, found {:?}", expected, self.peek()),
                position: self.peek_pos(),
            })
        }
    }

    fn parse_expr(&mut self) -> Result<Node, ExprError> {
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> Result<Node, ExprError> {
        let cond = self.parse_logical()?;
        if *self.peek() == Token::Question {
            self.advance();
            let then_branch = self.parse_expr()?;
            self.expect(&Token::Colon)?;
            let else_branch = self.parse_expr()?;
            Ok(Node::Ternary(
                Box::new(cond),
                Box::new(then_branch),
                Box::new(else_branch),
            ))
        } else {
            Ok(cond)
        }
    }

    fn parse_logical(&mut self) -> Result<Node, ExprError> {
        let mut left = self.parse_comparison()?;
        loop {
            let op = match self.peek() {
                Token::And => BinOp::And,
                Token::Or => BinOp::Or,
                _ => break,
            };
            self.advance();
            let right = self.parse_comparison()?;
            left = Node::BinaryOp(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Node, ExprError> {
        let left = self.parse_additive()?;
        let op = match self.peek() {
            Token::Gt => BinOp::Gt,
            Token::Gte => BinOp::Gte,
            Token::Lt => BinOp::Lt,
            Token::Lte => BinOp::Lte,
            Token::EqEq => BinOp::Eq,
            Token::BangEq => BinOp::Neq,
            _ => return Ok(left),
        };
        self.advance();
        let right = self.parse_additive()?;
        Ok(Node::BinaryOp(op, Box::new(left), Box::new(right)))
    }

    fn parse_additive(&mut self) -> Result<Node, ExprError> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = Node::BinaryOp(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Node, ExprError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Node::BinaryOp(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Node, ExprError> {
        match self.peek().clone() {
            Token::Minus => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(Node::UnaryNeg(Box::new(inner)))
            }
            Token::Bang => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(Node::UnaryNot(Box::new(inner)))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Node, ExprError> {
        match self.peek().clone() {
            Token::Number(n) => {
                self.advance();
                Ok(Node::Literal(ExprValue::Number(n)))
            }
            Token::Str(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Node::Literal(ExprValue::Str(s)))
            }
            Token::Ident(ref id) => {
                let id = id.clone();
                let pos = self.peek_pos();
                match id.as_str() {
                    "true" => {
                        self.advance();
                        Ok(Node::Literal(ExprValue::Bool(true)))
                    }
                    "false" => {
                        self.advance();
                        Ok(Node::Literal(ExprValue::Bool(false)))
                    }
                    "null" => {
                        self.advance();
                        Ok(Node::Literal(ExprValue::Null))
                    }
                    "datum" => {
                        self.advance();
                        self.parse_datum_access()
                    }
                    _ => {
                        self.advance();
                        if *self.peek() == Token::LParen {
                            self.advance(); // consume '('
                            let args = self.parse_call_args()?;
                            self.expect(&Token::RParen)?;
                            match resolve_builtin(&id) {
                                Some(arity) => {
                                    if args.len() != arity {
                                        return Err(ExprError {
                                            message: format!(
                                                "{id} expects {arity} argument{}, found {}",
                                                if arity == 1 { "" } else { "s" },
                                                args.len()
                                            ),
                                            position: pos,
                                        });
                                    }
                                    Ok(Node::Call(id, args))
                                }
                                None => Err(ExprError {
                                    message: format!(
                                        "unknown function '{id}'; supported: {}",
                                        supported_builtins()
                                    ),
                                    position: pos,
                                }),
                            }
                        } else {
                            Err(ExprError {
                                message: format!("unexpected identifier: '{id}'"),
                                position: pos,
                            })
                        }
                    }
                }
            }
            Token::LParen => {
                self.advance();
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            Token::LBracket => {
                self.advance();
                let mut elements = Vec::new();
                if *self.peek() != Token::RBracket {
                    loop {
                        elements.push(self.parse_expr()?);
                        if *self.peek() == Token::Comma {
                            self.advance();
                            continue;
                        }
                        break;
                    }
                }
                self.expect(&Token::RBracket)?;
                Ok(Node::Array(elements))
            }
            _ => Err(ExprError {
                message: format!("unexpected token: {:?}", self.peek()),
                position: self.peek_pos(),
            }),
        }
    }

    /// Parse a comma-separated argument list up to (not including) the
    /// closing `)`. Shared by call-site parsing; array-literal elements use
    /// the same comma-loop inline since they close on `]` instead.
    fn parse_call_args(&mut self) -> Result<Vec<Node>, ExprError> {
        let mut args = Vec::new();
        if *self.peek() != Token::RParen {
            loop {
                args.push(self.parse_expr()?);
                if *self.peek() == Token::Comma {
                    self.advance();
                    continue;
                }
                break;
            }
        }
        Ok(args)
    }

    fn parse_datum_access(&mut self) -> Result<Node, ExprError> {
        let mut fields = Vec::new();
        loop {
            match self.peek().clone() {
                Token::Dot => {
                    self.advance();
                    match self.peek().clone() {
                        Token::Ident(ref field) => {
                            let field = field.clone();
                            self.advance();
                            fields.push(field);
                        }
                        _ => {
                            return Err(ExprError {
                                message: "expected field name after 'datum.'".into(),
                                position: self.peek_pos(),
                            });
                        }
                    }
                }
                Token::LBracket => {
                    self.advance();
                    match self.peek().clone() {
                        Token::Str(ref field) => {
                            let field = field.clone();
                            self.advance();
                            self.expect(&Token::RBracket)?;
                            fields.push(field);
                        }
                        _ => {
                            return Err(ExprError {
                                message: "expected string inside datum[...]".into(),
                                position: self.peek_pos(),
                            });
                        }
                    }
                }
                _ => break,
            }
        }
        if fields.is_empty() {
            return Err(ExprError {
                message: "expected field access after 'datum'".into(),
                position: self.peek_pos(),
            });
        }
        Ok(Node::DatumAccess(fields))
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

impl Expr {
    /// Parse an expression string into an evaluable AST.
    pub fn parse(input: &str) -> Result<Self, ExprError> {
        let tokens = tokenize(input)?;
        let mut parser = Parser::new(tokens);
        let root = parser.parse_expr()?;

        // Ensure all input was consumed
        if *parser.peek() != Token::Eof {
            return Err(ExprError {
                message: format!("unexpected token after expression: {:?}", parser.peek()),
                position: parser.peek_pos(),
            });
        }

        Ok(Self { root })
    }

    /// Evaluate the expression for a single row. The `datum` closure provides
    /// field values by column name.
    pub fn eval(&self, datum: &dyn Fn(&str) -> ExprValue) -> ExprValue {
        eval_node(&self.root, datum)
    }

    /// Every column name this expression reads, in sorted order.
    ///
    /// Only the first segment of a `datum.a.b` path is reported, matching what
    /// [`Self::eval`] actually looks up (Arrow columns are flat). Callers use
    /// this to validate column dtypes **once per batch** instead of per row —
    /// a dtype is a schema property, so the check belongs before evaluation.
    pub(crate) fn field_refs(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        collect_field_refs(&self.root, &mut out);
        out
    }
}

fn collect_field_refs(node: &Node, out: &mut BTreeSet<String>) {
    match node {
        Node::Literal(_) => {}
        Node::DatumAccess(fields) => {
            out.insert(fields[0].clone());
        }
        Node::Array(elements) | Node::Call(_, elements) => {
            for e in elements {
                collect_field_refs(e, out);
            }
        }
        Node::UnaryNeg(inner) | Node::UnaryNot(inner) => collect_field_refs(inner, out),
        Node::BinaryOp(_, left, right) => {
            collect_field_refs(left, out);
            collect_field_refs(right, out);
        }
        Node::Ternary(cond, then, other) => {
            collect_field_refs(cond, out);
            collect_field_refs(then, out);
            collect_field_refs(other, out);
        }
    }
}

fn eval_node(node: &Node, datum: &dyn Fn(&str) -> ExprValue) -> ExprValue {
    match node {
        Node::Literal(v) => v.clone(),
        Node::Array(elements) => {
            ExprValue::Array(elements.iter().map(|e| eval_node(e, datum)).collect())
        }
        Node::Call(name, args) => eval_call(name, args, datum),
        Node::DatumAccess(fields) => {
            // For nested access like datum.a.b, we only support single-level
            // column access in Arrow (the first field). Additional segments
            // would require nested struct support — treat as the first field
            // for now, which matches Vega's flat-datum convention.
            datum(&fields[0])
        }
        Node::UnaryNeg(inner) => {
            let v = eval_node(inner, datum);
            ExprValue::Number(-v.as_number())
        }
        Node::UnaryNot(inner) => {
            let v = eval_node(inner, datum);
            ExprValue::Bool(!v.is_truthy())
        }
        Node::BinaryOp(op, left, right) => {
            // Short-circuit for logical operators
            match op {
                BinOp::And => {
                    let l = eval_node(left, datum);
                    if !l.is_truthy() {
                        return ExprValue::Bool(false);
                    }
                    let r = eval_node(right, datum);
                    return ExprValue::Bool(r.is_truthy());
                }
                BinOp::Or => {
                    let l = eval_node(left, datum);
                    if l.is_truthy() {
                        return ExprValue::Bool(true);
                    }
                    let r = eval_node(right, datum);
                    return ExprValue::Bool(r.is_truthy());
                }
                _ => {}
            }

            let l = eval_node(left, datum);
            let r = eval_node(right, datum);

            match op {
                BinOp::Add => eval_add(&l, &r),
                BinOp::Sub => ExprValue::Number(l.as_number() - r.as_number()),
                BinOp::Mul => ExprValue::Number(l.as_number() * r.as_number()),
                BinOp::Div => {
                    let denom = r.as_number();
                    if denom == 0.0 {
                        ExprValue::Number(f64::NAN)
                    } else {
                        ExprValue::Number(l.as_number() / denom)
                    }
                }
                BinOp::Mod => {
                    let denom = r.as_number();
                    if denom == 0.0 {
                        ExprValue::Number(f64::NAN)
                    } else {
                        ExprValue::Number(l.as_number() % denom)
                    }
                }
                BinOp::Gt | BinOp::Gte | BinOp::Lt | BinOp::Lte | BinOp::Eq | BinOp::Neq => {
                    eval_comparison(*op, &l, &r)
                }
                BinOp::And | BinOp::Or => unreachable!(),
            }
        }
        Node::Ternary(cond, then_branch, else_branch) => {
            let c = eval_node(cond, datum);
            if c.is_truthy() {
                eval_node(then_branch, datum)
            } else {
                eval_node(else_branch, datum)
            }
        }
    }
}

/// Dispatch a resolved builtin call. `Node::Call` is only ever constructed
/// for a name that `resolve_builtin` found in `BUILTINS` (see
/// `parse_primary`), and `find_builtin` reads the same table, so a lookup
/// miss here would mean the parser and evaluator disagree about the table's
/// contents — an invariant violation, not a reachable user input.
fn eval_call(name: &str, args: &[Node], datum: &dyn Fn(&str) -> ExprValue) -> ExprValue {
    match find_builtin(name) {
        Some(f) => f(args, datum),
        None => unreachable!("'{name}' passed parse-time resolution but has no table entry"),
    }
}

/// `indexof(haystack, needle)` — index of the first array element equal to
/// `needle`, or -1 if `haystack` is not an array or contains no match.
fn eval_indexof(args: &[Node], datum: &dyn Fn(&str) -> ExprValue) -> ExprValue {
    let haystack = eval_node(&args[0], datum);
    let needle = eval_node(&args[1], datum);
    let index = match haystack {
        ExprValue::Array(items) => items.iter().position(|item| *item == needle),
        _ => None,
    };
    ExprValue::Number(index.map(|i| i as f64).unwrap_or(-1.0))
}

fn eval_add(l: &ExprValue, r: &ExprValue) -> ExprValue {
    // String concatenation if either operand is a string
    match (l, r) {
        (ExprValue::Str(a), ExprValue::Str(b)) => ExprValue::Str(format!("{a}{b}")),
        (ExprValue::Str(a), _) => ExprValue::Str(format!("{a}{}", format_value(r))),
        (_, ExprValue::Str(b)) => ExprValue::Str(format!("{}{b}", format_value(l))),
        _ => ExprValue::Number(l.as_number() + r.as_number()),
    }
}

fn format_value(v: &ExprValue) -> String {
    match v {
        ExprValue::Number(n) => {
            if *n == (*n as i64) as f64 && n.is_finite() {
                format!("{}", *n as i64)
            } else {
                format!("{n}")
            }
        }
        ExprValue::Str(s) => s.clone(),
        ExprValue::Bool(b) => b.to_string(),
        ExprValue::Null => "null".to_string(),
        ExprValue::Array(items) => {
            let inner: Vec<String> = items.iter().map(format_value).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

fn eval_comparison(op: BinOp, l: &ExprValue, r: &ExprValue) -> ExprValue {
    // Null comparisons: null == null is true, null compared to anything else is false
    if matches!(l, ExprValue::Null) || matches!(r, ExprValue::Null) {
        let both_null = matches!(l, ExprValue::Null) && matches!(r, ExprValue::Null);
        return ExprValue::Bool(match op {
            BinOp::Eq => both_null,
            BinOp::Neq => !both_null,
            _ => false,
        });
    }

    // String comparison (lexicographic)
    if let (ExprValue::Str(a), ExprValue::Str(b)) = (l, r) {
        let ord = a.cmp(b);
        return ExprValue::Bool(match op {
            BinOp::Gt => ord == std::cmp::Ordering::Greater,
            BinOp::Gte => ord != std::cmp::Ordering::Less,
            BinOp::Lt => ord == std::cmp::Ordering::Less,
            BinOp::Lte => ord != std::cmp::Ordering::Greater,
            BinOp::Eq => ord == std::cmp::Ordering::Equal,
            BinOp::Neq => ord != std::cmp::Ordering::Equal,
            _ => false,
        });
    }

    // Numeric comparison
    let ln = l.as_number();
    let rn = r.as_number();
    ExprValue::Bool(match op {
        BinOp::Gt => ln > rn,
        BinOp::Gte => ln >= rn,
        BinOp::Lt => ln < rn,
        BinOp::Lte => ln <= rn,
        BinOp::Eq => ln == rn,
        BinOp::Neq => ln != rn,
        _ => false,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: evaluate an expression with a simple field map.
    fn eval_with(input: &str, fields: &[(&str, ExprValue)]) -> ExprValue {
        let expr = Expr::parse(input).unwrap();
        expr.eval(&|name: &str| {
            fields
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.clone())
                .unwrap_or(ExprValue::Null)
        })
    }

    fn assert_num(v: ExprValue, expected: f64) {
        match v {
            ExprValue::Number(n) => {
                assert!(
                    (n - expected).abs() < 1e-10,
                    "expected {expected}, got {n}"
                );
            }
            other => panic!("expected Number({expected}), got {other:?}"),
        }
    }

    fn assert_bool(v: ExprValue, expected: bool) {
        assert_eq!(v, ExprValue::Bool(expected), "expected Bool({expected})");
    }

    // 1. Basic arithmetic
    #[test]
    fn basic_arithmetic() {
        let result = eval_with(
            "datum.x + datum.y * 2",
            &[("x", ExprValue::Number(3.0)), ("y", ExprValue::Number(4.0))],
        );
        // x + y*2 = 3 + 8 = 11
        assert_num(result, 11.0);
    }

    #[test]
    fn arithmetic_subtraction_and_division() {
        let result = eval_with(
            "datum.a - datum.b / 2",
            &[("a", ExprValue::Number(10.0)), ("b", ExprValue::Number(4.0))],
        );
        // 10 - 4/2 = 10 - 2 = 8
        assert_num(result, 8.0);
    }

    // 2. Comparisons
    #[test]
    fn comparison_gte() {
        let result = eval_with("datum.age >= 18", &[("age", ExprValue::Number(21.0))]);
        assert_bool(result, true);

        let result = eval_with("datum.age >= 18", &[("age", ExprValue::Number(16.0))]);
        assert_bool(result, false);
    }

    #[test]
    fn comparison_lt() {
        let result = eval_with("datum.x < 5", &[("x", ExprValue::Number(3.0))]);
        assert_bool(result, true);
    }

    // 3. Logical operators
    #[test]
    fn logical_and_or() {
        let result = eval_with(
            "datum.x > 0 && datum.y < 100",
            &[("x", ExprValue::Number(5.0)), ("y", ExprValue::Number(50.0))],
        );
        assert_bool(result, true);

        let result = eval_with(
            "datum.x > 0 && datum.y < 100",
            &[
                ("x", ExprValue::Number(-1.0)),
                ("y", ExprValue::Number(50.0)),
            ],
        );
        assert_bool(result, false);
    }

    #[test]
    fn logical_short_circuit() {
        // && short-circuits: if left is false, right is not evaluated
        let result = eval_with(
            "datum.x > 10 && datum.y > 0",
            &[("x", ExprValue::Number(5.0)), ("y", ExprValue::Number(-1.0))],
        );
        assert_bool(result, false);

        // || short-circuits: if left is true, right is not evaluated
        let result = eval_with(
            "datum.x > 0 || datum.y > 0",
            &[("x", ExprValue::Number(5.0)), ("y", ExprValue::Number(-1.0))],
        );
        assert_bool(result, true);
    }

    // 4. Ternary
    #[test]
    fn ternary_positive() {
        let result = eval_with(
            "datum.x > 0 ? datum.x : -datum.x",
            &[("x", ExprValue::Number(5.0))],
        );
        assert_num(result, 5.0);
    }

    #[test]
    fn ternary_negative() {
        let result = eval_with(
            "datum.x > 0 ? datum.x : -datum.x",
            &[("x", ExprValue::Number(-3.0))],
        );
        assert_num(result, 3.0);
    }

    // 5. Bracket access
    #[test]
    fn bracket_access() {
        let result = eval_with(
            "datum[\"full name\"] == \"Alice\"",
            &[("full name", ExprValue::Str("Alice".into()))],
        );
        assert_bool(result, true);
    }

    #[test]
    fn bracket_access_not_equal() {
        let result = eval_with(
            "datum[\"full name\"] == \"Alice\"",
            &[("full name", ExprValue::Str("Bob".into()))],
        );
        assert_bool(result, false);
    }

    // 6. String comparisons
    #[test]
    fn string_neq() {
        let result = eval_with(
            "datum.name != \"Bob\"",
            &[("name", ExprValue::Str("Alice".into()))],
        );
        assert_bool(result, true);

        let result = eval_with(
            "datum.name != \"Bob\"",
            &[("name", ExprValue::Str("Bob".into()))],
        );
        assert_bool(result, false);
    }

    #[test]
    fn string_lexicographic_comparison() {
        let result = eval_with(
            "datum.name > \"Alice\"",
            &[("name", ExprValue::Str("Bob".into()))],
        );
        assert_bool(result, true);

        let result = eval_with(
            "datum.name < \"Bob\"",
            &[("name", ExprValue::Str("Alice".into()))],
        );
        assert_bool(result, true);
    }

    // 7. Division by zero
    #[test]
    fn division_by_zero_nan() {
        let result = eval_with(
            "datum.x / datum.y",
            &[("x", ExprValue::Number(10.0)), ("y", ExprValue::Number(0.0))],
        );
        match result {
            ExprValue::Number(n) => assert!(n.is_nan(), "expected NAN, got {n}"),
            other => panic!("expected Number(NAN), got {other:?}"),
        }
    }

    #[test]
    fn modulo_by_zero_nan() {
        let result = eval_with(
            "datum.x % 0",
            &[("x", ExprValue::Number(10.0))],
        );
        match result {
            ExprValue::Number(n) => assert!(n.is_nan(), "expected NAN, got {n}"),
            other => panic!("expected Number(NAN), got {other:?}"),
        }
    }

    // 8. Unary operators
    #[test]
    fn unary_neg() {
        let result = eval_with("-datum.x", &[("x", ExprValue::Number(7.0))]);
        assert_num(result, -7.0);
    }

    #[test]
    fn unary_not() {
        let result = eval_with("!datum.flag", &[("flag", ExprValue::Bool(true))]);
        assert_bool(result, false);

        let result = eval_with("!datum.flag", &[("flag", ExprValue::Bool(false))]);
        assert_bool(result, true);
    }

    // 9. Nested parentheses
    #[test]
    fn nested_parens() {
        let result = eval_with(
            "(datum.a + datum.b) * datum.c",
            &[
                ("a", ExprValue::Number(2.0)),
                ("b", ExprValue::Number(3.0)),
                ("c", ExprValue::Number(4.0)),
            ],
        );
        // (2+3)*4 = 20
        assert_num(result, 20.0);
    }

    #[test]
    fn deeply_nested_parens() {
        let result = eval_with(
            "((datum.a + datum.b) * (datum.c - datum.d))",
            &[
                ("a", ExprValue::Number(1.0)),
                ("b", ExprValue::Number(2.0)),
                ("c", ExprValue::Number(10.0)),
                ("d", ExprValue::Number(3.0)),
            ],
        );
        // (1+2)*(10-3) = 3*7 = 21
        assert_num(result, 21.0);
    }

    // 10. Invalid expressions
    #[test]
    fn invalid_import() {
        let err = Expr::parse("import os").unwrap_err();
        assert!(
            err.message.contains("unexpected identifier") || err.message.contains("function"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn invalid_function_call_unknown_name() {
        let err = Expr::parse("open()").unwrap_err();
        assert!(
            err.message.contains("unknown function 'open'") && err.message.contains("indexof"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn invalid_trailing_operator() {
        let err = Expr::parse("datum.x +").unwrap_err();
        assert!(err.position > 0, "error position should be non-zero");
    }

    #[test]
    fn invalid_empty_expr() {
        let err = Expr::parse("").unwrap_err();
        assert!(!err.message.is_empty());
    }

    // 11. Null handling
    #[test]
    fn null_arithmetic_propagates() {
        let result = eval_with(
            "datum.x + 1",
            &[("x", ExprValue::Null)],
        );
        // Null.as_number() -> NAN, NAN + 1 = NAN
        match result {
            ExprValue::Number(n) => assert!(n.is_nan()),
            other => panic!("expected Number(NAN), got {other:?}"),
        }
    }

    #[test]
    fn null_comparison() {
        let result = eval_with("datum.x == null", &[("x", ExprValue::Null)]);
        assert_bool(result, true);

        let result = eval_with("datum.x == null", &[("x", ExprValue::Number(5.0))]);
        assert_bool(result, false);
    }

    #[test]
    fn null_neq() {
        let result = eval_with("datum.x != null", &[("x", ExprValue::Number(5.0))]);
        assert_bool(result, true);
    }

    // Additional edge cases
    #[test]
    fn string_concatenation() {
        let result = eval_with(
            "datum.first + \" \" + datum.last",
            &[
                ("first", ExprValue::Str("John".into())),
                ("last", ExprValue::Str("Doe".into())),
            ],
        );
        assert_eq!(result, ExprValue::Str("John Doe".into()));
    }

    #[test]
    fn bool_literal_in_comparison() {
        let result = eval_with("datum.flag == true", &[("flag", ExprValue::Bool(true))]);
        assert_bool(result, true);
    }

    #[test]
    fn numeric_literal_expr() {
        let result = eval_with("datum.x * 0.5 + 1.5", &[("x", ExprValue::Number(4.0))]);
        assert_num(result, 3.5);
    }

    #[test]
    fn single_quoted_string() {
        let result = eval_with(
            "datum.name == 'Alice'",
            &[("name", ExprValue::Str("Alice".into()))],
        );
        assert_bool(result, true);
    }

    #[test]
    fn type_mismatch_returns_nan() {
        // "hello" - 5 -> NAN (string.as_number() = NAN)
        let result = eval_with(
            "datum.x - 5",
            &[("x", ExprValue::Str("hello".into()))],
        );
        match result {
            ExprValue::Number(n) => assert!(n.is_nan()),
            other => panic!("expected Number(NAN), got {other:?}"),
        }
    }

    // ─── Membership syntax: array literals + indexof (F-L05-06) ───────────

    #[test]
    fn membership_numeric_found_and_not_found() {
        let result = eval_with(
            "indexof([1, 2, 3], datum.id) >= 0",
            &[("id", ExprValue::Number(2.0))],
        );
        assert_bool(result, true);

        let result = eval_with(
            "indexof([1, 2, 3], datum.id) >= 0",
            &[("id", ExprValue::Number(5.0))],
        );
        assert_bool(result, false);
    }

    #[test]
    fn membership_string_found_and_not_found() {
        let result = eval_with(
            "indexof([\"US\", \"DE\", \"JP\"], datum.country) >= 0",
            &[("country", ExprValue::Str("DE".into()))],
        );
        assert_bool(result, true);

        let result = eval_with(
            "indexof([\"US\", \"DE\", \"JP\"], datum.country) >= 0",
            &[("country", ExprValue::Str("FR".into()))],
        );
        assert_bool(result, false);
    }

    #[test]
    fn indexof_returns_first_matching_index() {
        let result = eval_with("indexof([10, 20, 30], 20)", &[]);
        assert_num(result, 1.0);

        let result = eval_with("indexof([10, 20, 30], 99)", &[]);
        assert_num(result, -1.0);
    }

    #[test]
    fn indexof_empty_array_is_not_found() {
        let result = eval_with("indexof([], 1)", &[]);
        assert_num(result, -1.0);
    }

    #[test]
    fn indexof_mixed_dtype_array() {
        // Elements compare equal only within the same ExprValue variant —
        // a numeric needle never matches a string element and vice versa.
        let result = eval_with("indexof([1, \"two\", 3], \"two\")", &[]);
        assert_num(result, 1.0);

        let result = eval_with("indexof([1, \"two\", 3], 2)", &[]);
        assert_num(result, -1.0);

        let result = eval_with("indexof([1, \"two\", 3], 3)", &[]);
        assert_num(result, 2.0);
    }

    #[test]
    fn indexof_non_array_first_arg_is_not_found() {
        // datum.x is a scalar Number, not an Array — indexof treats a
        // non-array haystack as empty rather than panicking.
        let result = eval_with("indexof(datum.x, 1)", &[("x", ExprValue::Number(1.0))]);
        assert_num(result, -1.0);
    }

    #[test]
    fn indexof_arity_error_names_function_and_count() {
        let err = Expr::parse("indexof([1, 2])").unwrap_err();
        assert_eq!(err.message, "indexof expects 2 arguments, found 1");

        let err = Expr::parse("indexof([1, 2], 1, 3)").unwrap_err();
        assert_eq!(err.message, "indexof expects 2 arguments, found 3");
    }

    #[test]
    fn unknown_function_names_supported_set() {
        let err = Expr::parse("lookup([1, 2], 1)").unwrap_err();
        assert!(
            err.message.contains("unknown function 'lookup'") && err.message.contains("indexof"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn array_literal_with_expression_elements() {
        // "any expression elements" — not just literals.
        let result = eval_with(
            "indexof([datum.a + 1, datum.b * 2], 6)",
            &[("a", ExprValue::Number(1.0)), ("b", ExprValue::Number(3.0))],
        );
        assert_num(result, 1.0);
    }

    // ─── Previously-valid corpus regression sweep ──────────────────────────
    //
    // Every construct in the documented syntax table (plus the
    // implemented-but-undocumented forms: %, true/false/null, scientific
    // notation, chained-looking paths, datum bracket access) must stay
    // byte-stable after the grammar additions above. This is the RED-proof
    // target for the Comma-token removal check (see the removed-and-restored
    // Comma arm proof run alongside this suite).
    #[test]
    fn previously_valid_corpus_is_byte_stable() {
        assert_num(eval_with("datum.x + datum.y", &[("x", ExprValue::Number(2.0)), ("y", ExprValue::Number(3.0))]), 5.0);
        assert_num(eval_with("datum.x - datum.y", &[("x", ExprValue::Number(5.0)), ("y", ExprValue::Number(2.0))]), 3.0);
        assert_num(eval_with("datum.x * datum.y", &[("x", ExprValue::Number(4.0)), ("y", ExprValue::Number(2.0))]), 8.0);
        assert_num(eval_with("datum.x / datum.y", &[("x", ExprValue::Number(9.0)), ("y", ExprValue::Number(3.0))]), 3.0);
        assert_num(eval_with("datum.x % datum.y", &[("x", ExprValue::Number(10.0)), ("y", ExprValue::Number(3.0))]), 1.0);
        assert_bool(eval_with("datum.x > 1 && datum.y < 10", &[("x", ExprValue::Number(2.0)), ("y", ExprValue::Number(5.0))]), true);
        assert_bool(eval_with("datum.x == true", &[("x", ExprValue::Bool(true))]), true);
        assert_bool(eval_with("datum.x == false", &[("x", ExprValue::Bool(false))]), true);
        assert_bool(eval_with("datum.x == null", &[("x", ExprValue::Null)]), true);
        assert_num(eval_with("1e3 + 2.5E-1", &[]), 1000.25);
        assert_bool(
            eval_with(
                "datum[\"full name\"] == \"Alice\"",
                &[("full name", ExprValue::Str("Alice".into()))],
            ),
            true,
        );
        assert_num(
            eval_with(
                "datum.x > 0 ? datum.x : -datum.x",
                &[("x", ExprValue::Number(-4.0))],
            ),
            4.0,
        );
        assert_bool(
            eval_with(
                "datum.city == \"Zürich\"",
                &[("city", ExprValue::Str("Zürich".into()))],
            ),
            true,
        );
        // datum.a.b: only the first path segment is used (documented flat-
        // datum convention) — unaffected by the Comma/Array/Call additions.
        assert_bool(
            eval_with("datum.a.b == 1", &[("a", ExprValue::Number(1.0))]),
            true,
        );
    }

    // ─── Non-ASCII string literals (rust-quality Required Fix 1) ──────────

    #[test]
    fn non_ascii_string_literal_equality() {
        let result = eval_with(
            "datum.city == \"Zürich\"",
            &[("city", ExprValue::Str("Zürich".into()))],
        );
        assert_bool(result, true);

        let result = eval_with(
            "datum.city == \"Zürich\"",
            &[("city", ExprValue::Str("Paris".into()))],
        );
        assert_bool(result, false);
    }

    #[test]
    fn non_ascii_string_literal_membership() {
        let result = eval_with(
            "indexof([\"Zürich\", \"Paris\"], datum.city) >= 0",
            &[("city", ExprValue::Str("Zürich".into()))],
        );
        assert_bool(result, true);

        let result = eval_with(
            "indexof([\"Zürich\", \"Paris\"], datum.city) >= 0",
            &[("city", ExprValue::Str("Berlin".into()))],
        );
        assert_bool(result, false);
    }

    #[test]
    fn non_ascii_string_literal_after_escape() {
        // A backslash-escape immediately followed by a non-ASCII char (the
        // fallback arm of the escape match) must also decode the full char,
        // not just its lead byte.
        let result = eval_with(r#""\é" == "\é""#, &[]);
        assert_bool(result, true);
    }

    // ---- field_refs (batch-D: per-batch dtype validation needs the ref set) ----

    fn refs(src: &str) -> Vec<String> {
        Expr::parse(src).unwrap().field_refs().into_iter().collect()
    }

    #[test]
    fn field_refs_collects_from_every_node_kind() {
        // Binary + unary + ternary + array + call, all in one expression.
        let got = refs(
            "!datum.flag ? -datum.a + datum.b : indexof([datum.c, 2], datum.d)",
        );
        assert_eq!(got, vec!["a", "b", "c", "d", "flag"]);
    }

    #[test]
    fn field_refs_is_deduplicated_and_sorted() {
        assert_eq!(refs("datum.z > 1 && datum.a < datum.z"), vec!["a", "z"]);
    }

    #[test]
    fn field_refs_reports_only_the_first_path_segment() {
        // eval() looks up `fields[0]`; field_refs must agree with it.
        assert_eq!(refs("datum.a.b > 1"), vec!["a"]);
    }

    #[test]
    fn field_refs_empty_for_literal_only_expressions() {
        assert!(refs("1 + 2 > 2").is_empty());
    }
}
