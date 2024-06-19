import React, {useState} from "react"
import ReactDOM from 'react-dom'
import { Document, Page, pdfjs} from "react-pdf"
pdfjs.GlobalWorkerOptions.workerSrc = ''


function home(){

    const [file, setFile] = useState(null)

    function handlePDFUpload(){

    }

    return (
    <div className ="Home ">
        <h2> Enter a PDF you want to search through</h2>

        <input onChange = { ()=>{setFile.target.files[0]}}  type= "file"/>
 
        <Button>onClick = {handlePDFUpload}= Upload</Button>

    
 

    </div>
    )
}

export default home