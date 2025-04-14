# Q. How to handle or avoid a stack overflow in C++

In C++ a stack overflow usually leads to an unrecoverable crash of the program. For programs that need to be really robust, this is an unacceptable behaviour, particularly because stack size is limited. A few questions about how to handle the problem. Is there a way to prevent stack overflow by a general technique. (A scalable, robust solution, that includes dealing with external libraries eating a lot of stack, etc.) Is there a way to handle stack overflows in case they occur? Preferably, the stack gets unwound until there's a handler to deal with that kind of issue. There are languages out there, that have threads with expandable stacks. Is something like that possible in C++? Any other helpful comments on the solution of the C++ behavior would be appreciated.


# Q. Why does jQuery or a DOM method such as getElementById not find the element?

What are the possible reasons for document.getElementById, $("#id") or any other DOM method / jQuery selector not finding the elements? Example problems include: jQuery silently failing to bind an event handler


# Q. How can I prevent SQL injection in PHP?

If user input is inserted without modification into an SQL query, then the application becomes vulnerable to SQL injection, like in the following example:

$unsafe_variable = $_POST['user_input']; 

mysql_query("INSERT INTO `table` (`column`) VALUES ('$unsafe_variable')");
That's because the user can input something like value'); DROP TABLE table;--, and the query becomes:

INSERT INTO `table` (`column`) VALUES('value'); DROP TABLE table;--')
What can be done to prevent this from happening?

# Q. Event binding on dynamically created elements?

I have a bit of code where I am looping through all the select boxes on a page and binding a .hover event to them to do a bit of twiddling with their width on mouse on/off.

This happens on page ready and works just fine.

The problem I have is that any select boxes I add via Ajax or DOM after the initial loop won't have the event bound.

I have found this plugin (jQuery Live Query Plugin), but before I add another 5k to my pages with a plugin, I want to see if anyone knows a way to do this, either with jQuery directly or by another option.

# Q. Do I cast the result of malloc?

In this question, someone suggested in a comment that I should not cast the result of malloc. i.e., I should do this:

int *sieve = malloc(sizeof(*sieve) * length);
rather than:

int *sieve = (int *) malloc(sizeof(*sieve) * length);
Why would this be the case?

# Q. Adding scripting functionality to .NET applications
I have a little game written in Javascript. It uses a database as back-end. It's <a href="http://en.wikipedia.org/wiki/Collectible_card_game">trading card game</a>, and I wanted to implement the function of the cards as a script.