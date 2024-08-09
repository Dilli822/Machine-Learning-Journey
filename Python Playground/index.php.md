<?xml version="1.0" encoding="UTF-8" ?>
<School>

<class>
<grade> 1 </grade>
<students> 20 </students>
</class>

<class>
<grade> 2</grade>
<students> 25 </students>
</class>

</School>

DTD
<!DOCTYPE School [
<!ELEMENT School(class+)>
<!ELEMENT class(grade, students)>
<!ELEMENT grade (#PCDATA)>
<!ELEMENT students(#PCDATA)>
]>

XSD - defines the structure of the XML Document
<? xml version="1.0" ?>



