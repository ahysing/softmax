#include <memory>
#include <vector>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <iostream>

void putValuesFromStdin(std::weak_ptr<std::vector<double>> valuesPtr, bool verbose)
{
    struct termios tc = {};
    tcgetattr(STDIN_FILENO, &tc);
    auto c_lflag = tc.c_lflag;
    auto c_cc_vmin = tc.c_cc[VMIN];
    auto c_cc_vtime = tc.c_cc[VTIME];
    // non-canonical mode; no echo.
    tc.c_lflag &= ~ICANON;
    tc.c_cc[VMIN] = 0;   // bytes until read unblocks.
    tc.c_cc[VTIME] = 50; // timeout.
    tcsetattr(STDIN_FILENO, TCSANOW, &tc);

    size_t lineReadSize = 1024 * 1024 * 2;
    char lineRead[lineReadSize];
    memset(lineRead, '\0', lineReadSize);

    size_t sizeRead = read(STDIN_FILENO, lineRead, lineReadSize);
    if (verbose && sizeRead != 0)
    {
        std::cout << lineRead << std::endl;
        std::cout << std::endl;
    }
    
    tc.c_lflag = c_lflag;
    tc.c_cc[VMIN] = c_cc_vmin;
    tc.c_cc[VTIME] = c_cc_vtime;
    tcsetattr(STDIN_FILENO, TCSANOW, &tc);


    bool lineIsLetter[lineReadSize];
    memset(lineIsLetter, true, lineReadSize);
    for (unsigned int i = 0; i < lineReadSize; i++)
    {
        lineIsLetter[i] &= lineRead[i] >= '0'; 
    }

    for (unsigned int i = 0; i < lineReadSize; i++)
    {
        lineIsLetter[i] &= lineRead[i] <= '9'; 
    }

    for (unsigned int i = 0; i < lineReadSize; i++)
    {
        lineIsLetter[i] |= lineRead[i] == '.'; 
    }

    for (unsigned int i = 0; i < lineReadSize; i++)
    {
        lineIsLetter[i] |= lineRead[i] == ','; 
    }

    for (unsigned int i = 0; i < lineReadSize; i++)
    {
        lineIsLetter[i] |= lineRead[i] == 'e'; 
    }

    for (unsigned int i = 0; i < lineReadSize; i++)
    {
        lineIsLetter[i] |= lineRead[i] == '-'; 
    }

    auto values = valuesPtr.lock();
    if (values == nullptr)
    {
        std::cerr << "Failed loading weak_ptr. input is null." << std::endl;
        return;
    }

    unsigned int at = 0;
    while (at < sizeRead)
    {
        for (; lineIsLetter[at] == false && at < sizeRead; at ++) {}

        if (lineIsLetter[at])
        {
            char *parseError = nullptr;
            const char* currentArg = lineRead + at;
            double value = strtod(currentArg, &parseError);
            if (parseError != currentArg)
            {
                values->push_back(value);
            } else {
                std::cerr << "Failed parsing number " << currentArg << " at position " << at << std::endl;
            }

            for (; lineIsLetter[at] == true && at < sizeRead; at ++) {}
        }
    }
}
