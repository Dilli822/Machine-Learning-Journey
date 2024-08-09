<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "cct";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Handle form submission
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $id = $_POST['id'];
    $firstname = $_POST['firstname'];
    $lastname = $_POST['lastname'];
    $email = $_POST['email'];

    $sql = "UPDATE student SET firstname='$firstname', lastname='$lastname', email='$email' WHERE id=$id";

    if ($conn->query($sql) === TRUE) {
        echo "Record updated successfully";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

$conn->close();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Update Student Record</title>
</head>
<body>
    <h2>Update Student Record</h2>
    <form action="<?php echo $_SERVER['PHP_SELF']; ?>" method="post">
        <label for="id">Student ID:</label>
        <input type="number" id="id" name="id" required>
        <br>
        <label for="firstname">First Name:</label>
        <input type="text" id="firstname" name="firstname" required>
        <br>
        <label for="lastname">Last Name:</label>
        <input type="text" id="lastname" name="lastname" required>
        <br>
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>
        <br>
        <input type="submit" value="Update Record">
    </form>
</body>
</html>
