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

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $firstname = $_POST['firstname'];
    $lastname = $_POST['lastname'];
    $email = $_POST['email'];

    $sql = "INSERT INTO student (firstname, lastname, email) VALUES ('$firstname', '$lastname', '$email')";

    if ($conn->query($sql) === TRUE) {
        echo "New record created successfully";
        header('Location: retrive.php');
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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Create Student Record</title>
</head>
<body>
    <h2>Create New Student Record</h2>
    <form action="<?php echo $_SERVER['PHP_SELF']; ?>" method="post">
        <label for="firstName">First Name:</label>
        <input type="text" id="firstname" name="firstname" required>
        <br>
        <label for="lastName">Last Name:</label>
        <input type="text" id="lastname" name="lastname" required>
        <br>
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>
        <br>
        <input type="submit" value="Create Record">
    </form>
</body>
</html>