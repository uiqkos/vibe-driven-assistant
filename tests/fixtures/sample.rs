use std::fmt;
use std::collections::HashMap;

/// A trait for displayable items
pub trait Displayable {
    /// Format as string
    fn display(&self) -> String;
}

/// Represents a user
pub struct User {
    pub name: String,
    pub email: String,
    pub age: u32,
}

/// Status enum for users
pub enum Status {
    Active,
    Inactive,
    Banned(String),
}

impl User {
    /// Create a new user
    pub fn new(name: String, email: String, age: u32) -> Self {
        User { name, email, age }
    }

    /// Get user greeting
    pub fn greet(&self) -> String {
        format!("Hello, {}!", self.name)
    }
}

impl Displayable for User {
    fn display(&self) -> String {
        format!("{} <{}>", self.name, self.email)
    }
}

/// Format a status value
pub fn format_status(status: &Status) -> String {
    match status {
        Status::Active => "active".to_string(),
        Status::Inactive => "inactive".to_string(),
        Status::Banned(reason) => format!("banned: {}", reason),
    }
}
