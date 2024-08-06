// book catalog for the php CRUD Operations

<!DOCTYPE html>
<html lang="en">
<head>
<title> Book Catalog </title>
</head>
<body>

<form action="process.php" method="post">
<label for="searchType"> Book Catalog </label> <br>
<select id="searchType"> 
<option value="title">title </option>
<option value="autho">author </option>
</select>
<br>
<label for="download">Download </label>
<input type="radio" value="1" name="download"> Yes <br>
<input type="radio" value="0" name="download"> No
<br>
<label for="keyword">keyword </label>
<input type="text" required name="keyword" id="keyword">
<br>

<input type="submit" value="Search" >

</form>

