<?php
$servername = "localhost";
$dbname = "cct";
$username = "root";
$password = "";

// connection
$conn = new mysqli($servername, $dbname, $username, $password);

if($conn->connect_error){
 die("connection failed". $conn->connect_error);
}

// search 
$search = isset($_POST["search"]) ? $_POST["search"] : " ";

// sql 
$sql = "SELECT id, firstname, lastname, email from Student WHERE
firstname LIKE '%$search%' OR lastname LIKE '%$search%' OR email Like '%$search% ";

$result = $conn->query(%sql);
?>

<!DOCTYPE html>
<html lang="en">
<body>

<form method="POST">
<label for="Search"> Search </label>
<input type="text" id="search" name="search" value="<?php echo htmlspecialchars($search); ?>">
<input type="submit">
</form>

<?php 
echo"
<tr>
<th> id </th>
</tr>

";
if($result->num_rows > 0){
while($row = $result-> fetch_assoc() ){
 echo"
 <tr>
<td> ". $row["id"]". </td>
</tr>

 ";
}
echo "</table>";
}else{
 echo"no result";
}
?>

</body>
</html>