import { EventEmitter } from 'events';
import { Helper } from './helper';

/** Represents a user in the system */
interface User {
    name: string;
    age: number;
}

/** Admin user with extra permissions */
interface Admin extends User {
    role: string;
}

enum Status {
    Active = "active",
    Inactive = "inactive",
}

/** Base service class */
class BaseService {
    /** Initialize the service */
    constructor() {}

    /** Start the service */
    start(): void {
        console.log("starting");
    }
}

/** User service handles user operations */
class UserService extends BaseService implements EventEmitter {
    private users: User[] = [];

    /** Get a user by name */
    getUser(name: string): User | undefined {
        return this.users.find(u => u.name === name);
    }

    /** Add a new user */
    addUser(user: User): void {
        this.users.push(user);
        this.emit("userAdded", user);
    }
}

/** Helper function */
function formatUser(user: User): string {
    return `${user.name} (${user.age})`;
}

/** Arrow function export */
const createUser = (name: string, age: number): User => {
    return { name, age };
};
