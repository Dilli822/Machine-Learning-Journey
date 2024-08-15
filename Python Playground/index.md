<?php

class Fruit{
function set_name($name){
 $this->name = $name;
}

function get_name(){
 return $this->name;
}
}

$apple = new Fruit();
$apple->set_name("Chocolate");
$apple->get_name();
?>