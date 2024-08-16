

<?php 

$servername = "localhost";
$dbname = "cct";
$username = "root";
$password = "";

// create a connection
$conn = new mysqli($servername, $dbname, $username, $password);

// check the connection
if($conn->connect_error){
 die("Connection Failed".$conn->connect_error);
}
if($_SERVER["REQUEST_METHOD"] == "POST"){
 $id 

 $sql = "INSERT INTO Student (firstname, lastname, email)
}

$conn->close();
?>



