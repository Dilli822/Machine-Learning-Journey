<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "catalog";

$conn = new mysqli($servername, $username, $password, $dbname);
if(conn->connect_error){
die("connection failed " .conn->connect_error);
}

IF($_SERVER["REQUEST_METHOD"] == "POST"){
$firstname = $_POST["firstname"];
$lastname = $_POST["lastname"];
$email = $_POST["email"];

$sql = "INSERT INTO  student (firstname, lastname, email) VALUES ('$firstname', '$lastname', '$email');

if($conn->query($sql) === TRUE){

}else{
echo "error" . $sql . $conn->error;
}


}
$conn->close();

?>