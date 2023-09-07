import styles from "./Features.module.css";
import ImageUploader from "../components/ImageUploader";
import { useState } from "react";
import { IoReturnUpBack } from "react-icons/io5";
import axios from "axios";
import { Player } from "@lottiefiles/react-lottie-player";
import animationImage from "../assets/animationImage.json";

export default function Features() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileSelect = (file) => {
    setSelectedImage(file);
    setPrediction(null);
  };

  const handleSubmit = () => {
    setIsLoading(true);
    // Create form data
    const formData = new FormData();
    if (typeof selectedImage === "string") {
      formData.append("filepath", selectedImage.split("/").pop()); // Only send the filename
    } else {
      formData.append("file", selectedImage);
    }

    // Send image to server for prediction
    axios
      .post("http://localhost:8000/upload", formData)
      .then((response) => {
        console.log(response.data);

        setTimeout(() => {
          setIsLoading(false);
          setPrediction(response.data.message);
        }, 3500);
      })
      .catch((error) => {
        console.error("Error uploading image:", error);
        alert(error.response?.data?.error || "Error uploading image");
        setIsLoading(false);
      });
  };

  const handleCancel = () => {
    setSelectedImage(null);
    setPrediction(null);
  };
  return (
    <div className={styles.features}>
      <div className={styles.leftFlex}>
        <h1>Is it AI or Not Ah?!</h1>
        <p>
          Determine whether an image has been generated by artificial
          intelligence or a human
        </p>
        <div className={styles.sampleImages}>
          <img
            src="https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-02.jpg"
            alt="Sample 1"
            className={styles.image}
            onClick={() => {
              setSelectedImage(
                "https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-02.jpg"
              );
              setPrediction(null);
            }}
          />
          <img
            src="https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-05.jpg"
            alt="Sample 2"
            className={styles.image}
            onClick={() => {
              setSelectedImage(
                "https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-05.jpg"
              );
              setPrediction(null);
            }}
          />
          <img
            src="https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-07.jpg"
            alt="Sample 3"
            className={styles.image}
            onClick={() => {
              setSelectedImage(
                "https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-07.jpg"
              );
              setPrediction(null);
            }}
          />
          <img
            src="https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-08.jpg"
            alt="Sample 4"
            className={styles.image}
            onClick={() => {
              setSelectedImage(
                "https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-08.jpg"
              );
              setPrediction(null);
            }}
          />
          <img
            src="https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/img000004.jpg"
            alt="Sample 5"
            className={styles.image}
            onClick={() => {
              setSelectedImage(
                "https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/img000004.jpg"
              );
              setPrediction(null);
            }}
          />
          <img
            src="https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-09-04_23-49-00.jpg"
            alt="Sample 6"
            className={styles.image}
            onClick={() => {
              setSelectedImage(
                "https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-09-04_23-49-00.jpg"
              );
              setPrediction(null);
            }}
          />
          <img
            src="https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-09-04_23-44-33.jpg"
            alt="Sample 7"
            className={styles.image}
            onClick={() => {
              setSelectedImage(
                "https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-09-04_23-44-33.jpg"
              );
              setPrediction(null);
            }}
          />
          <img
            src="https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-16.jpg"
            alt="Sample 8"
            className={styles.image}
            onClick={() => {
              setSelectedImage(
                "https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-16.jpg"
              );
              setPrediction(null);
            }}
          />
          <img
            src="https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-18.jpg"
            alt="Sample 9"
            className={styles.image}
            onClick={() => {
              setSelectedImage(
                "https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/photo_2023-08-17_23-10-18.jpg"
              );
              setPrediction(null);
            }}
          />
          <img
            src="https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/img000026.jpg"
            onClick={() => {
              setSelectedImage(
                "https://raw.githubusercontent.com/jiaawe/Anaconda-AI-ML-competition/main/flask-server/static/img000026.jpg"
              );
              setPrediction(null);
            }}
            alt="Sample 10"
            className={styles.image}
          />
        </div>
      </div>
      <div className={styles.rightFlex}>
        {isLoading && (
          <Player
            autoplay
            loop
            src={animationImage}
            className={styles.lottieContainer}
          ></Player>
        )}
        {!isLoading &&
          (selectedImage ? (
            <div className={styles.imageWrapper}>
              <button onClick={handleCancel} className={styles.cancelButton}>
                <IoReturnUpBack size={20} />
              </button>
              <img
                src={
                  typeof selectedImage === "string"
                    ? selectedImage
                    : URL.createObjectURL(selectedImage)
                }
                alt="Uploaded Preview"
                className={styles.uploadedImage}
              />
            </div>
          ) : (
            <ImageUploader onFileSelect={handleFileSelect} />
          ))}

        <button
          className={styles.btn}
          onClick={handleSubmit}
          disabled={!selectedImage || isLoading}
        >
          Submit
        </button>
        {prediction && (
          <p
            className={`${styles.predictionResult} ${
              prediction === "Human Generated"
                ? styles.humanGenerated
                : styles.notHumanGenerated
            }`}
          >
            This image is {prediction}
          </p>
        )}
      </div>
    </div>
  );
}