<?php
// search and display the result 
$servername = "localhost";
$dbname = "cct";
$username = "root";
$password = "";

// connection
$conn = new mysqli($servername, $dbname, $username, $password);

// checking the connection
if($conn->connect_error){
 die("Connection failed". $conn->connect_error);
}

if($_SERVER['REQUEST_METHOD'] == "POST"){
$id = $_POST['id'];
$firstname = $_POST['firstname'];
$lastname = $_POST['lastname'];
$email = $_POST['email'];

$sql = "UPDATE student SET firstname = '$firstname',
lastname = '$lastname',
email = '$email' 
WHERE id = $id";




if($conn->query($sql)){
 echo"success updated";
}
else{
echo "error";
}

}

$conn->close();

?>

<form method = "POST" action="<?php echo $_SERVER['PHP_SELF']; ?>">