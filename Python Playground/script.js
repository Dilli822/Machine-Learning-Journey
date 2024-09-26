
// 1. object creation using object literals
let obj = {
    name: "dilli",
    age: 25
}

console.log(obj.name)
console.log(obj.age)

// 2. function 
function hello(age){
    this.age = age
}
let result = new hello(24);
console.log(result.age)

// 3. new Object keyword
let a = new Object();
a.name = "hello world";
console.log(a.name)

// 4. using class constructor
class Car{
    constructor(name, color){
        this.name = name;
        this.color = color;
    }
}

let r = new Car("volvo", "black")
console.log(r.color,
    r.name)



    let jsonOb = {
        "brand": "samsung",
        "price": 2323.2323,
        "pname": "Tab A9 4GB RAM/64GB ROM",
        "isOffer": false,
        "isStock": null
      };
      
      let jsonStr = JSON.stringify(jsonOb);
      console.log(jsonStr);
      
      console.log("json parse", JSON.parse(jsonStr))
      
      Object.entries(jsonOb).forEach(([key, value]) =>{
        console.log(`${key}: ${value}`);
      })