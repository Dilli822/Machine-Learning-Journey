
let JsonD = {
  name: "dilli",
  adress: "itahari-20"
}
Object.entries(JsonD).forEach(([key, value])=>{
 console.log(`${key} ${value}`)
})


let jsonStr = JSON.stringify(JsonD);
consol.elog(jsonStr);