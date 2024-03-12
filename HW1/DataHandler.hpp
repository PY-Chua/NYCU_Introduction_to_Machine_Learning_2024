#ifndef __DATA_HANDLER_HPP__
#define __DATA_HANDLER_HPP__

#include <sstream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace tools
{
    using Table = std::unordered_map<std::string, std::vector<float>>;

    // 将 Table 对象转换为字符串表示形式的函数
    std::string tableToString(const Table &table)
    {
        std::stringstream ss;
        ss << "{\n";
        for (const auto &entry : table)
        {
            ss << "  \"" << entry.first << "\": [";
            for (size_t i = 0; i < entry.second.size(); ++i)
            {
                ss << entry.second[i];
                if (i != entry.second.size() - 1)
                {
                    ss << ", ";
                }
            }
            ss << "]\n";
        }
        ss << "}\n";
        return ss.str();
    }

    std::string tableDetail(const Table &table)
    {
        std::string ss;

        for (const auto &entry : table)
        {
            ss += "(key:[" + entry.first + "], length : " + std::to_string(entry.second.size()) + ")\n";
        }

        ss += "\n";
        return ss;
    }

    std::vector<std::string> splitBy(const std::string &str, const char &op)
    {
        std::stringstream ss(str);
        std::vector<std::string> result = {""};
        while (std::getline(ss, result.back(), op))
        {
            result.push_back("");
        }
        result.pop_back();
        return result;
    }

    std::vector<float> mappingToFloat(const std::vector<std::string> &strList)
    {
        std::vector<float> result;
        std::transform(strList.begin(), strList.end(), std::back_inserter(result),
                       [](const std::string &str)
                       { return std::stof(str); });

        return result;
    }

    Table readCsv(const std::string &filename)
    {
        std::ifstream inFile;
        inFile.exceptions(std::ios::eofbit | std::ios::failbit | std::ios::badbit);
        std::stringstream ssIn;

        try
        {
            inFile.open(filename);
            ssIn << inFile.rdbuf();
            inFile.close();
        }
        catch (const std::exception &e)
        {
            throw e;
        }

        std::string headerStr;
        std::getline(ssIn, headerStr);

        auto headerStrList = splitBy(headerStr, ',');

        Table table;
        for (const auto &header : headerStrList)
        {
            // add new vector
            table[header] = {};
        }
        std::string temp;
        while (std::getline(ssIn, temp))
        {
            auto dataLine = splitBy(temp, ',');
            auto dataLineFloat = mappingToFloat(dataLine);

            for (size_t i = 0; i < headerStrList.size(); ++i)
            {
                table[headerStrList[i]].push_back(dataLineFloat[i]);
            }
        }

        return table;
    }

} // namespace tools

#endif