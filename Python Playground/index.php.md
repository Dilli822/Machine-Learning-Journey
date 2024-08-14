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

// search 
$search = isset($_POST['search']) ? $_POST['search'] : " ";

// sql query
$sql = "SELECT id, firstname, lastname, email FROM student 
WHERE firstname LIKE '%$firstname%' OR
lastname LIKE '%$lastname%' OR
email LIKE '%$email' ";

// result 
$result = $conn->query($sql);
?>

<form method="POST">
<input type="text" id="search" value="<?php echo htmlspecialchars($search); ?">
<input type="submit" value="Search">
</form>

<?php

if($result->num_rows > 0){
echo"<table>
<tr> 
<th> id </th>
<th> firstname </th>
<th> lastname </th>
<th> email </th>
</tr>
";
while($row = $result->fetch_assoc()){
echo"<tr>

<td> "$row['id']" </td>
</tr>";
}
echo"</table>";
}else{
echo "no data found";
}
?>