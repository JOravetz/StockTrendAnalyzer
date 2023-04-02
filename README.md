# Alpaca WebSocket C Program Example

## Summary
This C program, `alpaca_websocket_jansson.c`, connects to Alpaca's WebSocket API and subscribes to real-time trade, quote, and bar data for specified symbols. It prints the received data to the console. The program allows the user to choose between SIP or IEX data source. The program requires the APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables to be set, which are used for authentication.

## Installation and Compilation

To successfully compile and run the provided C program, you need to have the following libraries installed on your Ubuntu or Linux system:

1. libwebsockets - A lightweight WebSocket library.
2. jansson - A C library for encoding, decoding, and manipulating JSON data.

<pre>
sudo apt update
sudo apt install libwebsockets-dev libjansson-dev
</pre>

On other Linux distributions, use the respective package manager and package names to install the required libraries. For example, on Fedora or other RPM-based systems, you can use the dnf package manager:

To install these libraries on Ubuntu or other Debian-based systems, you can use the apt package manager. Open a terminal and run the following commands:

<pre>
sudo dnf install libwebsockets-devel jansson-devel
</pre>

Once you've installed the necessary libraries, you can proceed to compile and run the C program using the provided Makefile.

To compile the program, simply use the provided Makefile:

<pre>
make
</pre>

## Usage

<pre>
./alpaca_websocket_jansson [-t trades] [-q quotes] [-b bars] [-s sip]
</pre>

Options:
- `-t trades`: Comma-separated list of trade symbols, or `*` for all trades (with quotes).
- `-q quotes`: Comma-separated list of quote symbols, or `*` for all quotes (with quotes).
- `-b bars`: Comma-separated list of bar symbols, or `*` for all bars (with quotes).
- `-s sip`: Choose the data source. Allowed values are 'sip' (default) or 'iex'.

To exit the program, press Ctrl+C.

## How the main program works with the library and header file
The main program uses a library `alpaca_lib_jansson` and its corresponding header file `alpaca_lib_jansson.h`. The library provides reusable functions for parsing command-line options, handling WebSocket callbacks, and interacting with the Alpaca WebSocket API.

The header file includes function declarations, which allow the main program to call the functions defined in the library. By using the library and header file, the main program achieves modularity, making it easier to maintain, update, and reuse code.

## Code Explanation
The main program uses the alpaca_lib_jansson.h header file and the corresponding library, which contains the necessary functions to handle the connection and data processing.

The protocols structure is used for WebSocket communication with the Alpaca API, specifying the protocol name and callback function.

The main function starts by parsing the command-line options to set the subscription parameters (trade, quote, and bar symbols) and the data source (SIP or IEX).

The program sets up a WebSocket context and connection information, including the WebSocket server address and connection protocol. It then connects to the WebSocket server using the lws_client_connect_via_info() function.

The main event loop processes WebSocket events using the lws_service() function until the user interrupts the program with Ctrl+C. The program uses a SIGINT signal handler to detect this interruption and set the interrupted variable accordingly.

When the program is interrupted, it cleans up by destroying the WebSocket context and deleting the JSON object containing the subscription parameters.

## Good C Programming and Reusable Code

The program demonstrates good C programming practices and reusable code in several ways:

1. It uses a separate header file (alpaca_lib_jansson.h) and library to handle the Alpaca API connection and data processing. This modular approach makes it easier to reuse the code for other applications or extend the program's functionality.
2. It uses command-line options to allow users to specify the subscription parameters and data source, making the program flexible and easy to use for different purposes.
3. It employs error handling and informative messages to guide users and notify them of issues, such as incorrect command-line options or connection problems.
4. The program uses the widely-used Jansson library for JSON manipulation, ensuring compatibility and ease of integration with other projects.
5. The code follows a clean and consistent coding style, making it easy to read and understand.
