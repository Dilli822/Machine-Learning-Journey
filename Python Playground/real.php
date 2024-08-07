<?php

// 1. array with array keyword
$a = array("hello", "world", 2023);
$arr = ["hello", "world", 23232];

$arr = array (
"name" => "dilli"
);

$mua = array(
array("hello", "world", 23.23),
array("hello", 232.232, 23)
);

$mual = [ 
["hsd", "efe"],
[232, 232.232],
];

$muaa = array(
array("hello" => "sds")
);


$Product = array(
array("pcode" => 001, "pname" => "watch", "price" => 1212.22),
array("pcode" => 002, "pname" => "tv", "price" => 1222.22)
);

echo "

<html>
<body>

<table>
<tr>
<th>pcode </th>
<th>pname </th>
<th>price </th>
</tr>

";

foreach($Product as $items){
echo"<tr>
<td> {items['pcode']} </td>
</tr>
";
};

echo"
</table>
</body>
";
?>