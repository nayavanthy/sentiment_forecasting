import { useState } from "react";
import {
  ChakraProvider,
  Box,
  Input,
  Button,
  Text,
  Image,
  VStack,
  Spinner,
} from "@chakra-ui/react";

const customStyles = {
  control: (provided, state) => ({
    ...provided,
    backgroundColor: "black",
    borderColor: state.isFocused ? "#39FF14" : "white",
    boxShadow: state.isFocused ? "0 0 5px #39FF14" : null,
    "&:hover": {
      borderColor: "#39FF14",
    },
    color: "white",
  }),
  option: (provided, state) => ({
    ...provided,
    backgroundColor: state.isSelected ? "#39FF14" : "black",
    color: state.isSelected ? "black" : "white",
    "&:hover": {
      backgroundColor: "#39FF14",
      color: "black",
    },
  }),
  multiValue: (provided) => ({
    ...provided,
    backgroundColor: "#39FF14",
    color: "black",
  }),
  multiValueLabel: (provided) => ({
    ...provided,
    color: "black",
  }),
  multiValueRemove: (provided) => ({
    ...provided,
    color: "black",
    "&:hover": {
      backgroundColor: "white",
      color: "black",
    },
  }),
};

const App = () => {
  const [topic, setTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [keyword, setKeyword] = useState("");
  const [imageVisible, setImageVisible] = useState(false);

  const handleProcess = async () => {
    if (!topic.trim()) {
      alert("Please enter a topic.");
      return;
    }

    setLoading(true);
    setKeyword("");
    setImageVisible(false);

    try {
      const response = await fetch("http://127.0.0.1:8000/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic }),
      });

      if (!response.ok) {
        throw new Error("Processing failed");
      }

      console.log("Processing started...");

      setTimeout(fetchKeyword, 5000);
      checkIfDone();
    } catch (error) {
      console.error("Error:", error);
      alert("Error processing request.");
      setLoading(false);
    }
  };

  const fetchKeyword = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/keyword_static/keywords.txt");
      if (!response.ok) throw new Error("Keyword file not found");

      const text = await response.text();
      setKeyword(text);
    } catch (error) {
      console.error("Error fetching keyword.txt:", error);
    }
  };

  const checkIfDone = async () => {
    let attempts = 0;
    const interval = setInterval(async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/forecast_static/arima_sentiment_prediction.png");
        if (response.ok) {
          setImageVisible(true);
          setLoading(false);
          clearInterval(interval);
        }
      } catch (error) {
        console.error("Waiting for image...", error);
      }

      attempts++;
      if (attempts >= 20) {
        clearInterval(interval);
        setLoading(false);
      }
    }, 5000);
  };

  return (
    <Box textAlign="center" p={10} minH="100vh" backgroundColor="black" color="white">
      <VStack spacing={4}>
        <Input
          placeholder="Enter topic"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          width="300px"
          borderColor="#39FF14"
          focusBorderColor="#39FF14"
          color="white"
        />
        <Button
          backgroundColor="black"
          color="#39FF14"
          border="1px solid #39FF14"
          _hover={{ backgroundColor: "#39FF14", color: "black" }}
          onClick={handleProcess}
          isDisabled={loading}
        >
          Start Process
        </Button>

        {loading && (
          <>
            <Spinner size="xl" color="#39FF14" />
            <Text>Processing... Please wait</Text>
          </>
        )}

        {keyword && (
          <Box p={4} bg="black" borderRadius="md" border="1px solid #39FF14">
            <Text fontSize="lg" fontWeight="bold">Keyword Extracted:</Text>
            <Text>{keyword}</Text>
          </Box>
        )}

        {imageVisible && (
          <Box mt={4}>
            <Text fontSize="lg" fontWeight="bold">ARIMA Sentiment Prediction:</Text>
            <Image
              src="http://127.0.0.1:8000/forecast_static/arima_sentiment_prediction.png"
              alt="ARIMA Sentiment Prediction"
              maxW="100%" 
              maxH="80vh" 
              objectFit="contain"
              mt={2}
            />
          </Box>
        )}
      </VStack>
    </Box>
  );
};

export default App;