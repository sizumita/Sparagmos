//! Intrinsic registry: name -> implementation. Unknown names are an explicit error.
use crate::error::{RuntimeError, RuntimeResult};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unit,
    Bool(bool),
    I64(i64),
    U64(u64),
    I32(i32),
    U32(u32),
    F64(f64),
    Str(String),
}
impl Value {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I64(v) => Some(*v),
            Value::I32(v) => Some(*v as i64),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            Value::U32(v) => Some(*v as u64),
            _ => None,
        }
    }
}

pub type IntrinsicFn = fn(&[Value]) -> RuntimeResult<Value>;

pub struct IntrinsicRegistry {
    map: HashMap<String, IntrinsicFn>,
}
impl IntrinsicRegistry {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
    pub fn register(&mut self, name: impl Into<String>, f: IntrinsicFn) {
        self.map.insert(name.into(), f);
    }
    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
    pub fn names(&self) -> Vec<&str> {
        self.map.keys().map(|s| s.as_str()).collect()
    }
    pub fn call(&self, name: &str, args: &[Value]) -> RuntimeResult<Value> {
        let f = self
            .map
            .get(name)
            .ok_or_else(|| RuntimeError::UnknownIntrinsic(name.to_string()))?;
        f(args)
    }
}
impl Default for IntrinsicRegistry {
    fn default() -> Self {
        default_registry()
    }
}

fn two_i64(name: &str, args: &[Value]) -> RuntimeResult<(i64, i64)> {
    if args.len() != 2 {
        return Err(RuntimeError::InvalidIntrinsicArgs {
            name: name.into(),
            expected: "2 integer args".into(),
        });
    }
    Ok((
        args[0]
            .as_i64()
            .ok_or_else(|| RuntimeError::InvalidIntrinsicArgs {
                name: name.into(),
                expected: "integer".into(),
            })?,
        args[1]
            .as_i64()
            .ok_or_else(|| RuntimeError::InvalidIntrinsicArgs {
                name: name.into(),
                expected: "integer".into(),
            })?,
    ))
}
fn two_u64(name: &str, args: &[Value]) -> RuntimeResult<(u64, u64)> {
    if args.len() != 2 {
        return Err(RuntimeError::InvalidIntrinsicArgs {
            name: name.into(),
            expected: "2 unsigned args".into(),
        });
    }
    Ok((
        args[0]
            .as_u64()
            .ok_or_else(|| RuntimeError::InvalidIntrinsicArgs {
                name: name.into(),
                expected: "unsigned".into(),
            })?,
        args[1]
            .as_u64()
            .ok_or_else(|| RuntimeError::InvalidIntrinsicArgs {
                name: name.into(),
                expected: "unsigned".into(),
            })?,
    ))
}

pub fn default_registry() -> IntrinsicRegistry {
    let mut r = IntrinsicRegistry::new();
    r.register("saturating_add", |a| {
        let (x, y) = two_i64("saturating_add", a)?;
        Ok(Value::I64(x.saturating_add(y)))
    });
    r.register("saturating_sub", |a| {
        let (x, y) = two_i64("saturating_sub", a)?;
        Ok(Value::I64(x.saturating_sub(y)))
    });
    r.register("saturating_add_unsigned", |a| {
        let (x, y) = two_u64("saturating_add_unsigned", a)?;
        Ok(Value::U64(x.saturating_add(y)))
    });
    r.register("wrapping_add", |a| {
        let (x, y) = two_i64("wrapping_add", a)?;
        Ok(Value::I64(x.wrapping_add(y)))
    });
    r.register("wrapping_sub", |a| {
        let (x, y) = two_i64("wrapping_sub", a)?;
        Ok(Value::I64(x.wrapping_sub(y)))
    });
    r.register("wrapping_mul", |a| {
        let (x, y) = two_i64("wrapping_mul", a)?;
        Ok(Value::I64(x.wrapping_mul(y)))
    });
    r.register("checked_add", |a| {
        let (x, y) = two_i64("checked_add", a)?;
        match x.checked_add(y) {
            Some(v) => Ok(Value::I64(v)),
            None => Err(RuntimeError::Overflow { op: "checked_add" }),
        }
    });
    r.register("size_of", |a| {
        // arg: Value::Str(type-name) — mirrors core::intrinsics::size_of conceptually for primitives
        let name = match a {
            [Value::Str(s)] => s.as_str(),
            _ => {
                return Err(RuntimeError::InvalidIntrinsicArgs {
                    name: "size_of".into(),
                    expected: "one type-name string".into(),
                })
            }
        };
        let sz = match name {
            "bool" | "u8" | "i8" => 1,
            "u16" | "i16" => 2,
            "u32" | "i32" | "char" | "f32" => 4,
            "u64" | "i64" | "f64" | "usize" | "isize" => 8,
            "u128" | "i128" => 16,
            _ => return Err(RuntimeError::UnknownIntrinsic(format!("size_of<{name}>"))),
        };
        Ok(Value::U64(sz as u64))
    });
    r.register("align_of", |a| {
        let name = match a {
            [Value::Str(s)] => s.as_str(),
            _ => {
                return Err(RuntimeError::InvalidIntrinsicArgs {
                    name: "align_of".into(),
                    expected: "one type-name string".into(),
                })
            }
        };
        let al = match name {
            "bool" | "u8" | "i8" => 1,
            "u16" | "i16" => 2,
            "u32" | "i32" | "char" | "f32" => 4,
            "u64" | "i64" | "f64" | "usize" | "isize" => 8,
            "u128" | "i128" => 16,
            _ => return Err(RuntimeError::UnknownIntrinsic(format!("align_of<{name}>"))),
        };
        Ok(Value::U64(al as u64))
    });
    r.register("ctlz", |a| {
        if a.len() != 1 {
            return Err(RuntimeError::InvalidIntrinsicArgs {
                name: "ctlz".into(),
                expected: "1 arg".into(),
            });
        }
        let v = a[0]
            .as_u64()
            .ok_or_else(|| RuntimeError::InvalidIntrinsicArgs {
                name: "ctlz".into(),
                expected: "unsigned".into(),
            })?;
        Ok(Value::U32(v.leading_zeros()))
    });
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saturating_and_wrapping() {
        let r = default_registry();
        assert_eq!(
            r.call("saturating_add", &[Value::I64(i64::MAX), Value::I64(1)])
                .unwrap(),
            Value::I64(i64::MAX)
        );
        assert_eq!(
            r.call("wrapping_add", &[Value::I64(i64::MAX), Value::I64(1)])
                .unwrap(),
            Value::I64(i64::MIN)
        );
        assert_eq!(
            r.call("wrapping_mul", &[Value::I64(3), Value::I64(4)])
                .unwrap(),
            Value::I64(12)
        );
    }
    #[test]
    fn size_of_intrinsic() {
        let r = default_registry();
        assert_eq!(
            r.call("size_of", &[Value::Str("i64".into())]).unwrap(),
            Value::U64(8)
        );
        assert_eq!(
            r.call("align_of", &[Value::Str("u16".into())]).unwrap(),
            Value::U64(2)
        );
    }
    #[test]
    fn unknown_is_explicit_error() {
        let r = default_registry();
        match r.call("transmute", &[]) {
            Err(RuntimeError::UnknownIntrinsic(n)) => assert_eq!(n, "transmute"),
            other => panic!("unexpected {other:?}"),
        }
    }
    #[test]
    fn checked_add_overflow() {
        let r = default_registry();
        assert!(matches!(
            r.call("checked_add", &[Value::I64(i64::MAX), Value::I64(1)]),
            Err(RuntimeError::Overflow { .. })
        ));
    }
    #[test]
    fn custom_register() {
        let mut r = IntrinsicRegistry::new();
        r.register("answer", |_| Ok(Value::I64(42)));
        assert_eq!(r.call("answer", &[]).unwrap(), Value::I64(42));
        assert!(r.contains("answer"));
    }
}
