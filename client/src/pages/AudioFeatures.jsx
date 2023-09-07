import styles from "./Features.module.css";
import AudioUploader from "../components/AudioUploader";
import { useState, useEffect } from "react";
import { IoReturnUpBack } from "react-icons/io5";
import axios from "axios";
import { Player } from "@lottiefiles/react-lottie-player";
import animationAudio from "../assets/animationAudio.json";

export default function FeaturesVideo() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileSelect = (file) => {
    setSelectedVideo(file);
    setPrediction(null);
  };

  const handleSubmit = () => {
    setIsLoading(true);
    // Create form data
    const formData = new FormData();
    if (typeof selectedVideo === "string") {
      formData.append("filepath", selectedVideo);
    } else {
      formData.append("file", selectedVideo);
    }

    // Send video to server for prediction
    axios
      .post("http://localhost:8000/upload_audio", formData)
      .then((response) => {
        setTimeout(() => {
          setIsLoading(false);
          setPrediction(response.data.message);
        }, 2000);
      })
      .catch((error) => {
        console.error("Error uploading video:", error);
        alert(error.response?.data?.error || "Error uploading video");
      });
  };

  const handleCancel = () => {
    setSelectedVideo(null);
    setPrediction(null);
  };

  return (
    <div className={styles.features}>
      <div className={styles.leftFlex}>
        <h1>Is it AI or Not Ah?!</h1>
        <p>
          Determine whether the audio inside the video has been generated by
          artificial intelligence or a human.
        </p>
        <div className={styles.sampleAudios}>
          <video
            src="https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/ANACONDA-interview.mp4"
            autoplay
            className={styles.audio}
            onClick={() => {
              setSelectedVideo(
                "https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/ANACONDA-interview.mp4"
              );
              setPrediction(null);
            }}
          />
          <video
            src="https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/223be94c4a-cat_20230905_175349_freemium.mp4"
            autoplay
            className={styles.audio}
            onClick={() => {
              setSelectedVideo(
                "https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/223be94c4a-cat_20230905_175349_freemium.mp4"
              );
              setPrediction(null);
            }}
          />
          <video
            src="https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/0621a9a217-thank-you_20230905_182713_freemium.mp4
            "
            autoplay
            className={styles.audio}
            onClick={() => {
              setSelectedVideo(
                "https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/0621a9a217-thank-you_20230905_182713_freemium.mp4"
              );
              setPrediction(null);
            }}
          />
          <video
            src="https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/smartest-kid.mp4"
            autoplay
            className={styles.audio}
            onClick={() => {
              setSelectedVideo(
                "https://can-detect-or-not-ah.s3.ap-southeast-1.amazonaws.com/smartest-kid.mp4"
              );
              setPrediction(null);
            }}
          />
        </div>
      </div>
      <div className={styles.rightFlex}>
        {isLoading && (
          <Player
            autoplay
            loop
            src={animationAudio}
            className={styles.lottieContainer}
          ></Player>
        )}
        {!isLoading &&
          (selectedVideo ? (
            <div className={styles.imageWrapper}>
              <button
                onClick={handleCancel}
                className={styles.cancelButtonVideo}
              >
                <IoReturnUpBack size={20} />
              </button>
              <video
                src={
                  typeof selectedVideo === "string"
                    ? selectedVideo
                    : URL.createObjectURL(selectedVideo)
                }
                autoplay
                controls
                className={styles.uploadedImage}
              />
            </div>
          ) : (
            <AudioUploader onFileSelect={handleFileSelect} />
          ))}
        <button
          className={styles.btn}
          onClick={handleSubmit}
          disabled={!selectedVideo || isLoading}
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
            The audio in this video is {prediction}
          </p>
        )}
      </div>
    </div>
  );
}
