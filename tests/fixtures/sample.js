import { utils } from './utils';

/** Base class for animals */
class Animal {
    /** Create an animal */
    constructor(name) {
        this.name = name;
    }

    /** Make a sound */
    speak() {
        console.log(`${this.name} makes a noise.`);
    }
}

/** Dog extends Animal */
class Dog extends Animal {
    /** Make a dog sound */
    speak() {
        console.log(`${this.name} barks.`);
    }
}

/** Create a dog */
function createDog(name) {
    return new Dog(name);
}

/** Arrow function for greeting */
const greet = (name) => {
    return `Hello, ${name}!`;
};
