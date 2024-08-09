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

// Handle search query
$search = isset($_POST['search']) ? $_POST['search'] : "";

// SQL query to retrieve data based on search input
$sql = "SELECT id, firstname, lastname, email FROM student 
        WHERE firstname LIKE '%$search%' 
        OR lastname LIKE '%$search%' 
        OR email LIKE '%$search%'";
$result = $conn->query($sql);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Search Student Records</title>
</head>
<body>
    <h2>Search Student Records</h2>
    <form method="post">
        <label for="search">Search:</label>
        <input type="text" id="search" name="search" value="<?php echo htmlspecialchars($search); ?>">
        <input type="submit" value="Search">
    </form>
    <?php
    if ($result->num_rows > 0) {
        echo "<table border='1'>
                <tr>
                    <th>ID</th>
                    <th>First Name</th>
                    <th>Last Name</th>
                    <th>Email</th>
                </tr>";
        while ($row = $result->fetch_assoc()) {
            echo "<tr>
                    <td>" . $row["id"] . "</td>
                    <td>" . $row["firstname"] . "</td>
                    <td>" . $row["lastname"] . "</td>
                    <td>" . $row["email"] . "</td>
                </tr>";
        }
        echo "</table>";
    } else {
        echo "No results found";
    }
    $conn->close();
    ?>
</body>
</html>
