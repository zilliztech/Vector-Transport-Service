/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.seatunnel.translation.spark.source.scan;

import org.apache.seatunnel.api.source.SeaTunnelSource;
import org.apache.seatunnel.api.table.type.SeaTunnelRow;
import org.apache.seatunnel.translation.spark.execution.MultiTableManager;
import org.apache.seatunnel.translation.spark.source.partition.batch.SeaTunnelBatch;
import org.apache.seatunnel.translation.spark.source.partition.micro.SeaTunnelMicroBatch;

import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.streaming.MicroBatchStream;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

import java.util.Map;

public class SeaTunnelScan implements Scan {

    private final SeaTunnelSource<SeaTunnelRow, ?, ?> source;

    private final int parallelism;
    private final String jobId;

    private final CaseInsensitiveStringMap caseInsensitiveStringMap;

    private final MultiTableManager multiTableManager;

    public SeaTunnelScan(
            SeaTunnelSource<SeaTunnelRow, ?, ?> source,
            int parallelism,
            String jobId,
            CaseInsensitiveStringMap caseInsensitiveStringMap,
            MultiTableManager multiTableManager) {
        this.source = source;
        this.parallelism = parallelism;
        this.jobId = jobId;
        this.caseInsensitiveStringMap = caseInsensitiveStringMap;
        this.multiTableManager = multiTableManager;
    }

    @Override
    public StructType readSchema() {
        return multiTableManager.getTableSchema();
    }

    @Override
    public Batch toBatch() {
        Map<String, String> envOptions = caseInsensitiveStringMap.asCaseSensitiveMap();
        return new SeaTunnelBatch(source, parallelism, jobId, envOptions, multiTableManager);
    }

    @Override
    public MicroBatchStream toMicroBatchStream(String checkpointLocation) {
        return new SeaTunnelMicroBatch(
                source,
                parallelism,
                jobId,
                checkpointLocation,
                caseInsensitiveStringMap,
                multiTableManager);
    }
}
