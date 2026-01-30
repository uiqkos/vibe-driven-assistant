package main

import (
	"fmt"
	"strings"
)

// Reader defines a read interface
type Reader interface {
	Read(p []byte) (n int, err error)
}

// Writer defines a write interface
type Writer interface {
	Write(p []byte) (n int, err error)
}

// ReadWriter combines Reader and Writer
type ReadWriter interface {
	Reader
	Writer
}

// User represents a user in the system
type User struct {
	Name  string
	Email string
	Age   int
}

// Admin embeds User and adds Role
type Admin struct {
	User
	Role string
}

// NewUser creates a new User
func NewUser(name, email string, age int) *User {
	return &User{
		Name:  name,
		Email: email,
		Age:   age,
	}
}

// FullName returns the display name
func (u *User) FullName() string {
	return strings.Title(u.Name)
}

// Greet prints a greeting
func (u *User) Greet() {
	fmt.Printf("Hello, %s!\n", u.FullName())
}

// Promote creates an Admin from a User
func (u *User) Promote(role string) *Admin {
	return &Admin{User: *u, Role: role}
}
