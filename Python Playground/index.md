<?php

$servername = "localhost";
$username = "root";
$password = "";
$dbname = "student";

// create connection
$conn = new mysqli($servername, $username,$password, $dbname);

// checking the connection
if($conn->connect_error){
 die("connection failed". $conn->connect_error);
}

if($_SERVER['REQUEST_METHOD'] == 'POST'){
 $firstname = $_POST['firstname'];
 $sql = "INSERT INTO student ($firstname, $lastname, $email) 
   FROM ('$firstname', '$lastname', '$email')";
  if($conn->query($sql) === TRUE){
   echo "success";
  header("location: retreive.php");
 }else{
 echo"error" .$sql .$conn->error;
 }
}
$conn->close();
?>