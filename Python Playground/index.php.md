<?php 

$servername = "localhost";
$dbname = "book_catalog";
$username = "root";
$password = "";

//create a connection
$conn = new mysqli($dbname, $servename, $username, $password)

// check connection
if($conn->connect_error){
  die("Connection Failed ". $conn->connect_error)
}

$conn->close()
>